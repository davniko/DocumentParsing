"""Three fresh three-descendant controls for the cleaned nested wording context."""

import json
from pathlib import Path

from prepare_multirow_current_views import subset

from document_ocr.atomic import atomic_publish_json

ROOT = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")


def read(path):
    return json.loads(path.read_text())


def main():
    current = read(ROOT / "staged-production-config.json")
    contracts = read(Path(current["declared_packages"]["contracts"]["path"]))
    for number, old_number, prefix in [
        (40, 2, "06a7c8b2"),
        (41, 3, "a1087462"),
        (42, 30, "b2617901"),
    ]:
        old = ROOT / f"batch-{old_number:02}"
        config = read(old / "production-replay/config.json")
        sid = next(s for s in config["source_ids"] if s[4:12] == prefix)
        for key in ("ownership", "auxiliary"):
            config[key] = current[key]
        for key in ("numeric_certificates", "geography_certificates", "wording_certificates"):
            config["independent"][key] = current["independent"][key]
        config["declared_packages"]["contracts"] = current["declared_packages"]["contracts"]
        directory = ROOT / f"batch-{number:02}"
        config["output"] = str(directory / "campaign")
        config["concurrency"] = 2
        config = subset(config, [sid], directory / "selected")
        atomic_publish_json(directory / "prepared-config.json", config)
        atomic_publish_json(
            directory / "source-contracts.json",
            {"version": 1, "sources": {sid: contracts["sources"][sid]}},
        )
        print(json.dumps({"batch": number, "source": sid, "freshWordingControls": 3}), flush=True)


if __name__ == "__main__":
    main()
