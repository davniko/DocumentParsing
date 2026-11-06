"""Inventory ambiguous scalar wrapping in exact published R30 render evidence."""

import argparse
import json
import re
import resource
import time
from collections import Counter

from migrate_r30_synthetic import DATA, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main(name):
    started = time.perf_counter()
    manifest_path = DATA / "manifest.json"
    manifest = read(manifest_path)
    findings = []
    checked = 0
    for certificate_pin in manifest["syntheticBatches"]:
        certificate = read(verify_pin(certificate_pin))
        with verify_pin(certificate["records"]).open() as stream:
            for line in stream:
                row = json.loads(line)
                rendered = read(verify_pin(row["evidence"]["rendered"]))
                checked += 1
                for page in rendered["geometry"]["pages"]:
                    for block in page.get("blocks", []):
                        if (
                            re.fullmatch(r"[0-9]+(?:[.,][0-9]+)*", block["text"])
                            and len(block["physicalFragments"]) > 1
                        ):
                            findings.append(
                                {
                                    "id": row["documentId"],
                                    "source": row["sourceDocumentId"],
                                    "rendered": row["evidence"]["rendered"],
                                    "page": page["page"],
                                    "block": block,
                                }
                            )
    summary = {
        "checked": checked,
        "affectedDocuments": len({r["id"] for r in findings}),
        "affectedBlocks": len(findings),
        "blocksBySource": dict(Counter(r["source"][4:12] for r in findings)),
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    if read(manifest_path) != manifest:
        raise ValueError("publication changed during numeric-fragment audit")
    atomic_publish_json(
        OUT / f"{name}.json",
        {"manifest": pin(manifest_path), "summary": summary, "findings": findings},
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    main(parser.parse_args().name)
