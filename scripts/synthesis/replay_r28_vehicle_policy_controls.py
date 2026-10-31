"""Read-only exact replay of retained accepted controls after eligibility repair."""

import argparse
import json
import time
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import create_campaign


def run(config_path, controls_path, output):
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    controls = json.loads(controls_path.read_text())
    campaign = create_campaign(Path.cwd(), config)
    rows = []
    for control in controls:
        pin = control["candidate"]
        path = Path(pin["path"])
        if sha256_file(path) != pin["sha256"]:
            raise ValueError("accepted control changed before read-only replay")
        old = json.loads(path.read_text())
        authority = control["finalReplayAuthority"]
        if sha256_file(Path(authority["path"])) != authority["sha256"]:
            raise ValueError("final-authority replay evidence changed")
        expected = next(
            row
            for row in json.loads(Path(authority["path"]).read_text())["rows"]
            if row["sampleId"] == old["documentId"]
        )
        if expected["status"] != "PASS" or expected["candidateSha256"] != pin["sha256"]:
            raise ValueError("final replay does not certify this exact original candidate")
        try:
            current = campaign.render(old["sourceDocumentId"], old["variant"], persist=False)
            if digest(current) != expected["fullCandidateDigest"]:
                raise ValueError("final-authority candidate digest is not reproduced")
            if (current["joinedRawText"], current["target"]) != (
                old["joinedRawText"],
                old["target"],
            ):
                raise ValueError("accepted text or target changed during eligibility repair")
            rows.append({"sampleId": old["documentId"], "candidate": pin, "status": "IDENTICAL"})
        except Exception as error:
            rows.append(
                {
                    "sampleId": old["documentId"],
                    "candidate": pin,
                    "status": "HELD",
                    "error": str(error),
                }
            )
    result = {
        "status": "PASS" if rows and all(r["status"] == "IDENTICAL" for r in rows) else "HELD",
        "config": str(config_path),
        "configSha256": sha256_file(config_path),
        "records": rows,
        "seconds": time.perf_counter() - started,
        "providerCalls": 0,
        "historicalCandidatesChanged": False,
    }
    atomic_publish_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.controls, args.output)
