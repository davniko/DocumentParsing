"""Compile exact CAS+DOT/UN scalar NIOSH properties for eligible DG identities."""

import argparse
import html
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

from document_ocr.hashing import sha256_bytes, sha256_file
from document_ocr.synthesis.curated import save
from document_ocr.synthesis.curated_dg_flash import DgFlashPointRegistry
from document_ocr.synthesis.dangerous_goods_registry import DangerousGoodsHmtRecord
from document_ocr.synthesis.template_compiler.dangerous_goods_packaging import profile

INDEX = "https://www.cdc.gov/niosh/npg/npgdcas.html"
MAX_BYTES = 512 * 1024


class NioshCards(HTMLParser):
    """Read card values, never the nearby flammability-threshold explanation."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth, self.capture, self.pending = 0, None, None
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        if tag != "div":
            return
        self.depth += 1
        classes = dict(attrs).get("class", "").split()
        kind = "title" if "card-title" in classes else "text" if "card-text" in classes else None
        if kind is not None:
            self.capture = [kind, self.depth, []]

    def handle_data(self, data):
        if self.capture is not None:
            self.capture[2].append(data)

    def handle_endtag(self, tag):
        if tag != "div":
            return
        if self.capture is not None and self.capture[1] == self.depth:
            kind, _, parts = self.capture
            value = " ".join("".join(parts).split())
            if kind == "title":
                self.pending = value
            elif self.pending in {"CAS No.", "DOT ID & Guide", "Flash Point"}:
                if self.pending in self.fields:
                    raise ValueError("duplicate NIOSH property card")
                self.fields[self.pending] = value
                self.pending = None
            self.capture = None
        self.depth -= 1


def scalar_flash_point(payload: str) -> tuple[dict, Decimal | None]:
    parser = NioshCards()
    parser.feed(payload)
    if set(parser.fields) != {"CAS No.", "DOT ID & Guide", "Flash Point"}:
        raise ValueError("NIOSH CAS/DOT/flash-point card structure changed")
    literal = parser.fields["Flash Point"].replace("\u2212", "-")
    match = re.fullmatch(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*°F", literal)
    return parser.fields, Decimal(match[1]) if match else None


def acquire(url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    with urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise ValueError(f"NIOSH fetch failed: {url}/{response.status}")
        payload = response.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("NIOSH source exceeds the explicit acquisition bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmt", type=Path, required=True)
    parser.add_argument("--ecics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 8:
        raise ValueError("NIOSH acquisition concurrency must be 1-8")
    out = args.output.resolve()
    index = acquire(INDEX, out / "sources/cas-index.html")
    urls = {}
    for row in re.findall(r"<tr\b[^>]*>.*?</tr>", index.decode(), re.S | re.I):
        cas = set(re.findall(r"\b\d{2,7}-\d{2}-\d\b", row))
        refs = set(re.findall(r"href=['\"]([^'\"]*npgd\d{4}\.html)", row, re.I))
        if len(cas) == len(refs) == 1:
            urls.setdefault(cas.pop(), set()).add(urljoin(INDEX, refs.pop()))
    if not urls:
        raise ValueError("NIOSH CAS index contains no individually indexed compounds")
    hmt = {
        r.record_id: r for r in map(DangerousGoodsHmtRecord.model_validate_json, args.hmt.open())
    }
    links = [json.loads(line) for line in args.ecics.open()]
    candidates, rejected = [], []
    for link in links:
        if link["disposition"] != "eligible_unique_maritime_hmt" or len(link["cas_numbers"]) != 1:
            continue
        record = hmt[link["hmt_record_id"]]
        try:
            profile(record)
        except ValueError:
            continue
        cas = link["cas_numbers"][0]
        if cas in urls and len(urls[cas]) == 1:
            candidates.append((record, cas, next(iter(urls[cas]))))
    if len({r.record_id for r, _, _ in candidates}) != len(candidates):
        raise ValueError("multiple CAS facts address one HMT identity")
    requested = sorted({url for _, _, url in candidates})

    def one(url):
        payload = acquire(url, out / "sources" / url.rsplit("/", 1)[-1])
        return url, payload

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pages = dict(pool.map(one, requested))
    records = []
    for record, cas, url in candidates:
        payload = pages[url]
        fields, value = scalar_flash_point(payload.decode())
        if fields["CAS No."] != cas:
            if re.findall(r"\b[0-9]{2,7}-[0-9]{2}-[0-9]\b", fields["CAS No."]) == [cas]:
                rejected.append(
                    {
                        "record": record.record_id,
                        "reason": "qualified_identity",
                        "url": url,
                        "literal": fields["CAS No."],
                    }
                )
                continue
            raise ValueError(f"NIOSH index/page CAS mismatch: {cas}/{fields['CAS No.']}/{url}")
        if record.un_number not in re.findall(r"\b[0-9]{4}\b", fields["DOT ID & Guide"]):
            rejected.append(
                {"record": record.record_id, "reason": "DOT_UN_not_supported", "url": url}
            )
            continue
        if value is None:
            rejected.append(
                {
                    "record": record.record_id,
                    "reason": "not_scalar_closed_cup",
                    "url": url,
                    "literal": fields["Flash Point"],
                }
            )
            continue
        title = re.search(
            r'<h1[^>]*id=[\'"]content[\'"][^>]*>(.*?)</h1>', payload.decode(), re.S | re.I
        )
        if title is None:
            raise ValueError("NIOSH chemical title is absent")
        records.append(
            {
                "hmt_record_id": record.record_id,
                "cas_number": cas,
                "un_number": record.un_number,
                "chemical_name": html.unescape(re.sub(r"<[^>]*>", "", title[1])),
                "value_fahrenheit": str(value),
                "source_url": url,
                "source_sha256": sha256_bytes(payload),
                "source_literal": fields["Flash Point"],
                "method": "closed_cup",
            }
        )
    registry = DgFlashPointRegistry.model_validate(
        {
            "schema_version": 1,
            "source_index_url": INDEX,
            "source_index_sha256": sha256_bytes(index),
            "hmt_sha256": sha256_file(args.hmt),
            "ecics_sha256": sha256_file(args.ecics),
            "records": sorted(records, key=lambda r: r["hmt_record_id"]),
        }
    )
    save(out / "registry.json", registry.model_dump(mode="json"))
    report = {
        "requestedPages": len(requested),
        "matchedChemicalRecords": len(candidates),
        "accepted": len(records),
        "rejected": rejected,
        "rejectionCounts": dict(Counter(r["reason"] for r in rejected)),
        "sourceBytes": sum(map(len, pages.values())) + len(index),
        "registrySha256": sha256_file(out / "registry.json"),
        "apiCostUSD": 0,
    }
    save(out / "compilation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
