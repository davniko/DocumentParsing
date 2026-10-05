"""Read-only compact source/PDF inspection of unused real-source candidates."""
import argparse
import json
import re
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from inspect_real_v7_topup import LEGAL
from run_real_v7_batch import ROOT, SOURCE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefixes", nargs="+")
    p.add_argument("--render", action="store_true")
    p.add_argument("--page", type=int, default=1)
    p.add_argument("--ocr-max-page", type=int)
    args = p.parse_args()
    inventory = json.loads((ROOT / "artifacts/kie-labeling/batch006-screen-20261005/novelty-inventory-v2.json").read_text())["documents"]
    sources = {r["documentId"]: r for r in map(json.loads, SOURCE.read_text().splitlines())}
    panels = []
    for prefix in args.prefixes:
        hits = [r for r in inventory if r["documentId"].startswith("doc_" + prefix)]
        assert len(hits) == 1, (prefix, len(hits))
        row = hits[0]
        print("SOURCE", prefix, row["pdf"], row["exclusions"])
        if args.render:
            with pdfium.PdfDocument(ROOT / row["pdf"]) as pdf:
                page = pdf[args.page - 1]
                pic = page.render(scale=1.7).to_pil().convert("RGB")
                panel = Image.new("RGB", (pic.width, pic.height + 24), "white")
                panel.paste(pic, (0, 24))
                ImageDraw.Draw(panel).text((4, 4), f"{prefix} page {args.page}", fill="red")
                panels.append(panel)
        else:
            current_page = 1
            for n, line in enumerate(sources[row["documentId"]]["joinedRawText"].splitlines(), 1):
                marker = re.fullmatch(r"--- PAGE (\d+) ---", line.strip())
                if marker:
                    current_page = int(marker[1])
                if args.ocr_max_page and current_page > args.ocr_max_page:
                    continue
                if line.strip() and not (len(line) > 300 and LEGAL.match(line)):
                    print(n, line)
    if panels:
        sheet = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
        x = 0
        for panel in panels:
            sheet.paste(panel, (x, 0))
            x += panel.width
        out = ROOT / "artifacts/kie-labeling/batch006-screen-20261005/reserve-pdf-review"
        out.mkdir(exist_ok=True)
        path = out / ("-".join(args.prefixes) + f"-p{args.page}.png")
        sheet.save(path)
        print(path)


if __name__ == "__main__":
    main()
