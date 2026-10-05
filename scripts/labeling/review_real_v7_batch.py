"""Read-only source/candidate inspection for a selected real-labeling batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def candidate(batch, row):
    folder = batch / "runs" / row["name"] / "refine"
    state = read(folder / "status.json")
    path = folder / ("target.json" if state["target"] is not None else "reviewed-draft.json")
    value = read(path)
    if state["target"] is not None:
        assert value == state["target"]
    return path, value, state


def show(batch, numbers, phase):
    for row in read(batch / "selection.json")["documents"]:
        if row["index"] not in numbers:
            continue
        print("\nDOCUMENT", row["name"], row["split"], "PDF", row["pdf"])
        if phase != "target":
            for n, line in enumerate((ROOT / row["input"] / "ocr.txt").read_text().splitlines(), 1):
                if line.strip():
                    print(n, line)
        if phase == "source":
            continue
        path, target, state = candidate(batch, row)
        print("TARGET", path, "STATUS", state["status"])
        for key, value in target["documentPatch"].items():
            if value is not None:
                print(key, json.dumps(value, ensure_ascii=False))
        print(
            "FINDINGS",
            json.dumps(
                {k: r for k, r in state["reviews"].items() if r["findings"]}, ensure_ascii=False
            ),
        )


def render(batch, number, pages):
    import pypdfium2 as pdfium

    row = next(r for r in read(batch / "selection.json")["documents"] if r["index"] == number)
    folder = batch / "manual-review/pdf"
    folder.mkdir(parents=True, exist_ok=True)
    with pdfium.PdfDocument(ROOT / row["pdf"]) as pdf:
        for number in pages:
            assert 1 <= number <= len(pdf)
            dest = folder / f"{row['name']}-{number}.png"
            if not dest.exists():
                page = pdf[number - 1]
                bitmap = page.render(scale=1.6)
                bitmap.to_pil().save(dest)
                bitmap.close()
                page.close()
            print(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--phase", choices=("source", "target", "final"), default="final")
    parser.add_argument("--pages", type=int, nargs="+")
    parser.add_argument("numbers", type=int, nargs="+")
    args = parser.parse_args()
    if args.pages:
        assert len(args.numbers) == 1
        render(ROOT / args.batch, args.numbers[0], args.pages)
    else:
        show(ROOT / args.batch, args.numbers, args.phase)
