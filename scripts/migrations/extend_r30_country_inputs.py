"""Build a separately pinned input-only country vocabulary from reviewed aliases."""
import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.country_presentations import Registry

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/registries/compiled/country-presentations-iso-cldr-48.2.3-v1"
OUT = ROOT / "artifacts/paddle-migration/r30-independent-audit-20261103/country-input-registry"
REVIEW = ROOT / "docs/analysis/r30-independent-audit-20261103/country-input-additions.json"


def main():
    registry = Registry.load(ROOT)
    manifest = json.loads((BASE / "production-manifest.json").read_text())
    review = json.loads(REVIEW.read_text())
    if review["status"] != "MAIN_AGENT_VERIFIED_INPUT_ALIASES":
        raise ValueError("country extension needs reviewed source authority")
    aliases = json.loads((BASE / "input-aliases.json").read_text())
    updated = copy.deepcopy(aliases)
    known = {r["surface"].upper() for r in aliases["aliases"]}
    for row in review["aliases"]:
        registry.preferred_name(row["code"])
        if row["surface"].upper() in known or not row["provenance"]["url"]:
            raise ValueError("alias must be new and have reviewed authority")
        updated["aliases"].append(row)
        known.add(row["surface"].upper())
    for filename, pin in manifest["files"].items():
        if sha256_file(BASE / filename) != pin["sha256"]:
            raise ValueError("base registry changed")
        if filename == "input-aliases.json":
            atomic_publish_json(OUT / filename, updated)
        else:
            atomic_publish_bytes(OUT / filename, (BASE / filename).read_bytes())
        pin["sha256"] = sha256_file(OUT / filename)
    manifest["inputAliasCount"] = len(updated["aliases"])
    manifest["reviewedExtension"] = {"path": str(REVIEW.relative_to(ROOT)), "sha256": sha256_file(REVIEW)}
    atomic_publish_json(OUT / "production-manifest.json", manifest)
    digest = sha256_file(OUT / "production-manifest.json")
    extended = Registry(OUT, expected_manifest_sha256=digest)
    for row in review["aliases"]:
        if extended.resolve(row["surface"]) != row["code"]:
            raise ValueError("new alias does not resolve correctly")
    for row in aliases["aliases"]:
        if extended.resolve(row["surface"]) != registry.resolve(row["surface"]):
            raise ValueError("extension changed an existing identity")
    print(json.dumps({"path": str((OUT / "production-manifest.json").relative_to(ROOT)), "sha256": digest}))


if __name__ == "__main__":
    main()
