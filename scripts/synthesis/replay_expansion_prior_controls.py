"""Read-only regression replay; changed thermal requests require fresh evidence.

Never patch historical requests to bypass current native-evidence validation.
The exclusion is recorded and must be covered by fresh thermal synthesis.
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from refresh_r28_multirow import subset_config
from document_ocr.atomic import atomic_publish_json

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "docs/analysis/r28-multirow-refresh1500-20261031/final-authority-carryover"
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101/authority-carryover/nonthermal"


def execute(job):
    with Path(job["output"]).with_suffix(".log").open("x") as stream:
        result = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "scripts/synthesis/replay_r28_vehicle_policy_controls.py",
             "--config", job["config"], "--controls", job["controls"], "--output", job["output"]],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
        )
    print(Path(job["output"]).parent.name, result.returncode, flush=True)
    return result.returncode


def main():
    jobs, held = [], []
    for old in json.loads((PREVIOUS / "jobs.json").read_text()):
        config_path = Path(old["config"])
        config = json.loads(config_path.read_text())
        requests = json.loads((ROOT / config["independent"]["requests"]["path"]).read_text())
        thermal = {k.rsplit(":", 1)[0] for k, v in requests.items() if v["regime"] in {"frozen", "chilled"}}
        controls = json.loads(Path(old["controls"]).read_text())
        keep = []
        for control in controls:
            candidate = json.loads((ROOT / control["candidate"]["path"]).read_text())
            if candidate["sourceDocumentId"] in thermal:
                held.append({"sourceDocumentId": candidate["sourceDocumentId"], "candidate": control["candidate"],
                             "reason": "Old frozen request has no native thermal-profile evidence; replaced by fresh thermal-domain synthesis, not replayed via a fallback."})
            else:
                keep.append(control)
        sources = [s for s in config["source_ids"] if s not in thermal]
        if not sources:
            continue
        directory = OUT / config_path.parent.name
        path = subset_config(config_path, sources, directory)
        controls_path = directory / "controls.json"
        atomic_publish_json(controls_path, keep)
        jobs.append({"config": str(path), "controls": str(controls_path), "output": str(directory / "result.json")})
    atomic_publish_json(OUT / "jobs.json", jobs)
    atomic_publish_json(OUT / "thermal-requires-fresh-evidence.json", held)
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(execute, jobs))
    if any(statuses):
        raise SystemExit("One or more replay jobs failed; inspect logs, no acceptance implied.")


if __name__ == "__main__":
    main()
