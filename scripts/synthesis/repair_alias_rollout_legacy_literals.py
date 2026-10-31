"""Remove two reviewed stale non-target geography literals in current R28.

No package alias migration: the auxiliary free-zone context becomes destination-
relative, and the garbled source OCR country suffix is omitted without adding a
country target. Both edits shorten isolated regions; centroids remain centered.
"""

import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json, atomic_write_bytes, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_reflow_policy import FontMetrics


def main():
    work = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/package-alias-rollout-20261030")
    evidence = work / "legacy-literal-repair"
    path = Path("data/curated/mpci-bl-r28-positions-v1/synthetic-train.jsonl")
    original = path.read_bytes()
    config = json.loads((work / "inspection/inventory.json").read_text())["config"]
    font = config["positions"]["reflow"]
    metrics = FontMetrics(Path(font["font_path"]), font["font_sha256"])
    assert metrics.length("DESTINATION FREE ZONE") < metrics.length("PORT SAID WEST FREE ZONE")
    rows, changes, counts = [], [], {"78e8864a": 0, "dd8": 0}
    for line in original.decode().splitlines(keepends=True):
        row = json.loads(line)
        prefix = row.get("sourceDocumentId", "")[4:]
        key = next((p for p in counts if prefix.startswith(p)), None)
        if key is None:
            rows.append(line)
            continue
        before = copy.deepcopy(row)
        old, new = (
            ("PORT SAID WEST FREE ZONE", "DESTINATION FREE ZONE")
            if key == "78e8864a"
            else (",EGYFI", "")
        )
        if old not in row["joinedRawText"]:
            assert old not in row["positionedText"]
            rows.append(line)
            continue
        for field in ("joinedRawText", "positionedText"):
            assert row[field].count(old) == 1
            if key == "dd8":
                assert "PORT OF DISCHARGE FROM VESSEL (12)" in row[field]
            row[field] = row[field].replace(old, new)
            row[field + "Sha256"] = digest(row[field].encode())
        assert row["target"] == before["target"]
        assert [x.rsplit(" ||", 1)[-1] for x in row["positionedText"].splitlines()] == [
            x.rsplit(" ||", 1)[-1] for x in before["positionedText"].splitlines()
        ]
        changes.append({"before": before, "after": row, "reason": key})
        counts[key] += 1
        rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    assert counts == {"78e8864a": 6, "dd8": 6}
    payload = "".join(rows).encode()
    atomic_publish_json(
        evidence / "reviewed-patch.json",
        {
            "status": "PASS",
            "counts": counts,
            "records": changes,
            "targetsAndCoordinateSuffixesUnchanged": True,
            "geometry": "centered shortening inside existing regions; no new line or glyph growth",
            "beforeSha256": sha256_file(path),
            "afterSha256": digest(payload),
        },
    )
    atomic_publish_json(
        evidence / "reviewed-decisions.json",
        {
            "78e8864a": (
                "Non-target free-zone context must not retain an unrelated source port. "
                "Destination-relative wording retains its role without asserting a "
                "particular free-zone name."
            ),
            "dd8": (
                "Source OCR suffix EGYFI is not an interpretable country alias; source "
                "target has no discharge country. Remove the stale suffix, preserve "
                "locality and country absence."
            ),
        },
    )
    atomic_write_bytes(evidence / "before/synthetic-train.jsonl", original)
    manifest_path = path.parent / "manifest.json"
    atomic_write_bytes(evidence / "before/manifest.json", manifest_path.read_bytes())
    manifest = json.loads(manifest_path.read_text())
    assert manifest["files"][path.name] == digest(original)
    atomic_write_bytes(path, payload)
    manifest["files"][path.name] = sha256_file(path)
    manifest["aliasRolloutLegacyLiterals20261030"] = {
        "counts": counts,
        "evidence": str(evidence / "reviewed-patch.json"),
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({"status": "PASS", "counts": counts, "sha256": sha256_file(path)}))


if __name__ == "__main__":
    main()
