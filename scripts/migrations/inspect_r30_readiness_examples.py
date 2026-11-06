"""Local diagnostic views of actual published records, not document-image synthesis."""

import hashlib
import json
from pathlib import Path

from audit_r30_training_readiness import OUT, SPLITS, compact, features
from PIL import Image, ImageDraw, ImageFont

DATA = Path("data/curated/mpci-bl-r30-paddle-centroids-v1")


def plot_distributions():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    measures = (
        ("input_tokens", "Input tokens, pinned trainer tokenizer"),
        ("target_tokens", "Target tokens including EOS"),
        ("height_median", "Median line height, normalized 0-1000"),
        ("density_median", "Median text density: characters x height / width"),
    )
    for split, color in zip(SPLITS, ("#16736d", "#bd5732", "#6254a4"), strict=True):
        records = json.loads((OUT / f"r30-{split}.json").read_text())["records"]
        for ax, (field, title) in zip(axes.flat, measures, strict=True):
            values = np.sort([r[field] if field in r else r["geometry"][field] for r in records])
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=split, color=color)
            ax.set_xlabel(title)
            ax.set_ylabel("Fraction of documents")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("R30 real / synthetic / validation input and geometry distributions")
    fig.tight_layout()
    fig.savefig(OUT / "distributions.png", dpi=150)
    plt.close(fig)


def page_plot(row, page_number, path):
    page = row["pages"][page_number]
    scale = 1500 / page["height"]
    im = Image.new("RGB", (round(page["width"] * scale) + 30, 1550), "white")
    draw = ImageDraw.Draw(im)
    draw.text((10, 8), row["documentId"] + f" p{page_number + 1}", fill="black")
    ids = [compact(c["equipmentIdentifier"]) for c in row["target"].get("containerInformation", [])]
    for r in page["regions"]:
        x0, y0, x1, y1 = [v * scale for v in r["box"]]
        y0 += 30
        y1 += 30
        color = "#b91c1c" if any(i in compact(r["text"]) for i in ids) else "#6b7280"
        draw.rectangle((x0, y0, x1, y1), outline=color, width=1)
        h = max(6, int((y1 - y0) * 0.66))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", h)
        while font.getlength(r["text"]) > x1 - x0 and h > 4:
            h -= 1
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", h)
        draw.text((x0, y0), r["text"], font=font, fill="black")
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        draw.ellipse((cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5), fill="#2563eb")
    im.save(path)


def main():
    output = OUT / "examples"
    output.mkdir(parents=True, exist_ok=True)
    chosen, real, reasons = {}, {}, {}
    # Specialty cases are deterministic; random controls are chosen by a separate hash.
    slots = (
        "thermal_setpoint",
        "multi_dg",
        "mixed_multirow",
        "equal_shared_prefix_8plus",
        "forwarding_agent",
    )
    for split in SPLITS:
        rows = []
        seen = set()
        for line in (DATA / f"{split}.jsonl").open():
            r = json.loads(line)
            if split == "real-train":
                real[r["documentId"]] = r
            f = features(r["target"])
            for slot in slots:
                if f[slot] and slot not in seen:
                    seen.add(slot)
                    chosen[r["documentId"]] = r
                    reasons.setdefault(r["documentId"], []).append(f"{split}/{slot}")
            if split == "synthetic-train":
                rows.append(
                    (
                        hashlib.sha256(
                            ("readiness-20261106" + r["documentId"]).encode()
                        ).hexdigest(),
                        r,
                    )
                )
                rows.sort(key=lambda x: x[0])
                del rows[6:]
            if r["documentId"].startswith(("doc_30116fcc", "doc_5cabcd4b")):
                chosen[r["documentId"]] = r
                reasons.setdefault(r["documentId"], []).append(
                    "OCR noise / long validation control"
                )
        for _, r in rows:
            chosen[r["documentId"]] = r
            reasons.setdefault(r["documentId"], []).append("hash-selected synthetic control")
    for r in list(chosen.values()):
        p = r["sourceDocumentId"]
        if p in real and p not in chosen:
            chosen[p] = real[p]
            reasons[p] = ["matched anonymized parent"]
    manifest = []
    for identity, r in chosen.items():
        name = identity[:20]
        (output / f"{name}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
        (output / f"{name}.txt").write_text(
            r["positionedText"]
            + "\n\nTARGET\n"
            + json.dumps(r["target"], indent=2, ensure_ascii=False)
        )
        page_plot(r, 0, output / f"{name}-page1.png")
        manifest.append(
            {
                "id": identity,
                "parent": r["sourceDocumentId"],
                "reasons": reasons[identity],
                "record": str(output / f"{name}.json"),
                "plot": str(output / f"{name}-page1.png"),
            }
        )
    (OUT / "example-inventory.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} examples saved; plots are layout diagnostics, not original PDFs.")


if __name__ == "__main__":
    main()
    plot_distributions()
