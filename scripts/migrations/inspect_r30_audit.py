"""Read-only evidence viewer for the independent R30 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/paddle-migration/r30-independent-audit-20261103"


def records():
    for split in ("real-train", "validation"):
        with (ROOT / "data/curated/mpci-bl-r30-paddle-centroids-v1" / (split + ".jsonl")).open() as stream:
            for line in stream:
                yield split, json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix")
    parser.add_argument("--regions", help="Comma-separated zero-based page:start-end spans")
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--pdf-page", type=int)
    parser.add_argument("--template", action="store_true", help="Inspect the independently reviewed synthetic control")
    args = parser.parse_args()
    review = None
    if args.template:
        reviews = [json.loads(p.read_text()) for p in (ART / "template-reviews-v2").glob("*.json")]
        chosen = [r for r in reviews if r["sourceDocumentId"][4:].startswith(args.prefix)]
        if len(chosen) != 1:
            raise ValueError("prefix must resolve to exactly one reviewed template")
        review = chosen[0]
        controls = json.loads((ROOT / "docs/analysis/r30-full-rebase-20261102/handoff-controls.json").read_text())
        candidates = [json.loads((ROOT / p["path"]).read_text()) for p in controls[review["sourceDocumentId"]]]
        matches = [("template", r) for r in candidates if r["documentId"] == review["id"]]
    else:
        matches = [(s, r) for s, r in records() if r["documentId"][4:].startswith(args.prefix)]
    if len(matches) != 1:
        raise ValueError("prefix must resolve to exactly one published source")
    split, row = matches[0]
    print(split, row["documentId"])
    for pointer in args.target:
        value = row["target"]
        for key in pointer.strip("/").split("/") if pointer.strip("/") else []:
            value = value[int(key)] if isinstance(value, list) else value.get(key)
            if value is None:
                break
        print(pointer, json.dumps(value, ensure_ascii=False))
    if args.regions:
        refs = set()
        for span in args.regions.split(","):
            page, bounds = span.split(":")
            lo, hi = map(int, bounds.split("-"))
            refs.update((int(page), i) for i in range(lo, hi + 1))
    else:
        if review is None:
            review = json.loads((ART / "source-reviews-v2" / (row["documentId"] + ".json")).read_text())
        refs = {
            (q["page"], i)
            for f in review["review"]["findings"]
            for q in f["evidence"]
            for i in range(max(0, q["region"] - 2), q["region"] + 3)
        }
    for pi, ri in sorted(refs):
        if ri < len(row["pages"][pi]["regions"]):
            r = row["pages"][pi]["regions"][ri]
            print(pi, ri, [round(v, 1) for v in r["box"]], r["text"])
    if args.pdf_page is not None:
        import pypdfium2

        source_id = review["sourceDocumentId"] if args.template else row["documentId"]
        source = json.loads(
            (ROOT / "artifacts/paddle-migration/r30-full-rebase-20261102/sources" / (source_id + ".json")).read_text()
        )
        path = ROOT / source["source"]["pdf"]
        output = Path("/tmp") / ("r30-audit-" + args.prefix + "-" + str(args.pdf_page) + ".png")
        with pypdfium2.PdfDocument(str(path)) as pdf:
            page = pdf[args.pdf_page]
            bitmap = page.render(scale=1800 / page.get_height())
            bitmap.to_pil().save(output)
        print("PDF image:", output)


if __name__ == "__main__":
    main()
