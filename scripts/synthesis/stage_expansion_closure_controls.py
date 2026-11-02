"""Replace explicitly stale historical evidence with fresh one/two/three-row controls."""

import json
from pathlib import Path

from expand_r28_2500 import selected_config

from document_ocr.atomic import atomic_publish_json

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/kie-synthesis-production/r28-expansion2500-20261101"
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"


def main():
    base = json.loads(
        (
            ROOT
            / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/expansion-20261101-v21"
            / "complete-source-authority-config.json"
        ).read_text()
    )
    stale = {}
    for path in sorted((OUT / "authority-carryover/nonthermal").glob("*/result.json")):
        for row in json.loads(path.read_text())["records"]:
            if row["status"] != "IDENTICAL":
                candidate = json.loads((ROOT / row["candidate"]["path"]).read_text())
                stale.setdefault(candidate["sourceDocumentId"], []).append(row)
    # This source already has freshly certified one/two/three-row controls.
    completed = json.loads(
        (OUT / "authority-controls/drum-owner-controls/acceptance.json").read_text()
    )
    sources = sorted(set(stale) - set(completed["sources"]))
    for prefix in ("1444835f", "0951955d"):
        matches = [s for s in base["source_ids"] if s.startswith("doc_" + prefix)]
        if len(matches) != 1:
            raise ValueError("thermal representative is not unique")
        sources.extend(matches)
    path = RUN / "closure-controls/config.json"
    config = selected_config(
        base,
        dict.fromkeys(sources, 3),
        path.parent / "campaign",
        2026110199,
        group_counts=(1, 2, 3),
    )
    if any(s in config["declared_packages"]["source_group_counts"] for s in sources):
        raise ValueError("selected control has a source-specific group restriction")
    atomic_publish_json(path, config)
    atomic_publish_json(
        OUT / "authority-carryover/closure-plan.json",
        {
            "status": "FRESH_CONTROLS_REQUIRED_NOT_ACCEPTED",
            "staleHistoricalEvidence": stale,
            "alreadyCertified": list(completed["sources"]),
            "sources": sources,
            "samples": len(sources) * 3,
            "config": str(path.relative_to(ROOT)),
            "scope": (
                "Fresh current-source synthesis replaces stale receipts, with additional "
                "frozen/chilled one/two/three-row controls. Historical evidence remains "
                "unchanged. Controls are not training additions."
            ),
        },
    )
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
