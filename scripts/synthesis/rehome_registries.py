"""One-time, byte-verified relocation of sampling registries and repository paths."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

MOVES = {
    "data/registries/": "artifacts/registries/sources/",
    "artifacts/kie-synthesis/registries/": "artifacts/registries/compiled/",
    "artifacts/kie-synthesis-production/registries/": "artifacts/registries/phrases/",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    receipt_path = root / "artifacts/registries/relocation-receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for row in receipt["files"]:
            if sha(root / row["newPath"]) != row["sha256"]:
                raise ValueError(f"relocated registry changed: {row['newPath']}")
        print(f"Verified {len(receipt['files'])} previously relocated files")
        return
    rows = []
    for old, new in MOVES.items():
        source, dest = root / old, root / new
        if not source.is_dir() or dest.exists():
            raise ValueError(f"ambiguous relocation: {source} -> {dest}")
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"registry symlink requires review: {path}")
            if path.is_file():
                rows.append(
                    {
                        "oldPath": str(path.relative_to(root)),
                        "newPath": new + str(path.relative_to(source)),
                        "sha256": sha(path),
                        "bytes": path.stat().st_size,
                    }
                )
    for old, new in MOVES.items():
        dest = root / new
        dest.parent.mkdir(parents=True, exist_ok=True)
        (root / old).rename(dest)
    for row in rows:
        if sha(root / row["newPath"]) != row["sha256"]:
            raise ValueError(f"relocation byte mismatch: {row['newPath']}")
    # Mechanical path rewrite only. Saved experimental receipts remain historical.
    paths = subprocess.check_output(
        ["rg", "--files", "configs", "src", "tests", "scripts"], cwd=root, text=True
    ).splitlines()
    changed = []
    for relative in paths:
        path = root / relative
        if path == Path(__file__).resolve() or path.suffix not in {".py", ".yaml", ".json", ".md"}:
            continue
        before = path.read_text()
        after = before
        for old, new in MOVES.items():
            after = after.replace(old, new)
        if before != after:
            path.write_text(after)
            changed.append(relative)
    receipt_path.write_text(
        json.dumps({"files": rows, "updatedPathReferences": changed}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "files": len(rows),
                "bytes": sum(r["bytes"] for r in rows),
                "pathReferencesUpdated": len(changed),
            }
        )
    )


if __name__ == "__main__":
    main()
