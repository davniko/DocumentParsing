"""Compile the specifically inspected non-adjacent package presentations.

This list records source decisions, not heuristic inference: only the named
owned quantity/private clause is extended; no intervening caption is consumed.
"""

import json
from pathlib import Path

from prepare_multirow_batch import compile_batch

from document_ocr.synthesis.independent_campaign import _complete_optional_package_suffix
from document_ocr.synthesis.independent_preparation import _load_source_context
from document_ocr.synthesis.independent_wording import prepare_blueprint

REVIEWED = {
    "0b7f3731": [("v011", 0, "1")],
    "0ede63c2": [("vehicle_count", 0, "1")],
    "157fadd2": [
        ("held-numeric:1", 0, "5 PALLET\n188 CARTON(S) IN 5 PALLET(S)\n188 CTNS IN 5 PALLET")
    ],
    "20cf0a0c": [("v006", 1, "1")],
    "2763b891": [("v007", 0, "1")],
    "37c96354": [
        ("v011", 0, "20"),
        ("anchor:documentPatch.cargoPackages[0].typeCategory", 0, "PACKAGES"),
    ],
    "3dc8551d": [("packages", n, "Ninety Nine") for n in range(4)],
    "4289d168": [("package_count", 1, "1")],
    "458d2fa0": [("packages", 0, "1")],
    "4ae247ca": [("row_package_0", 0, "22"), ("row_package_1", 0, "14")],
    "6cc8c295": [("packages", 1, "1")],
    "819af5e4": [("v010", 0, "1.800")],
    "aa69d7fe": [("package_count", 0, "1")],
    "b2617901": [("v034", 0, "3,930")],
    "b8544221": [("v022", 0, "400")],
    "baf80643": [("declared_package_count", 0, "2")],
    "c1f612a2": [("package_count", 1, "1")],
    "cdc8d998": [
        ("agent:package:0", 0, "252"),
        ("anchor:documentPatch.cargoPackages[0].typeCategory", 0, "PACKAGES"),
    ],
    "d274be39": [("package_count", 1, "1")],
    "dfaddb83": [("held-numeric:0", 0, "1,000x20KG BAGS\n(20 HEAT TREATED PALLETS)")],
    "e9275395": [("shipment_packages", 2, "5")],
}


def main():
    path = Path("configs/synthesis/mpci_bl_curated_v7_package_aliases.json")
    config = json.loads(path.read_text())
    context = _load_source_context(Path.cwd(), config)
    reviewed = {}
    for sid in config["source_ids"]:
        prefix = sid[4:12]
        if prefix not in REVIEWED:
            continue
        bp = context.numeric.apply_blueprint(context.source.blueprint(sid), sid)
        if sid in context.wording:
            bp = prepare_blueprint(bp, context.wording[sid])
        bp = _complete_optional_package_suffix(bp)
        regions = {(r.key, r.occurrence): r for r in bp.regions}
        owners = []
        selected = []
        for key, occurrence, expected in REVIEWED[prefix]:
            region = regions[key, occurrence]
            if region.source != expected:
                raise ValueError(f"reviewed source changed: {prefix}/{key}/{occurrence}")
            selected.append(region)
        groups = [selected] if prefix in {"37c96354", "cdc8d998"} else [[r] for r in selected]
        for index, group in enumerate(groups):
            start, end = min(r.start for r in group), max(r.end for r in group)
            inside = [r for r in bp.regions if r.start < end and start < r.end]
            if set(inside) != set(group):
                raise ValueError("reviewed package phrase crosses unrelated ownership")
            row = {
                "owners": [[r.key, r.occurrence] for r in group],
                "start": start,
                "end": end,
                "source": bp.source.encode()[start:end].decode(),
                "scope": index if prefix == "4ae247ca" else "shipment",
            }
            if prefix == "37c96354":
                if row["source"] != "20.00 PACKAGES":
                    raise ValueError("reviewed decimal-count phrase changed")
                row["quantityFormat"] = "decimal2"
            if prefix == "3dc8551d":
                row["quantityFormat"] = "words"
            if prefix in {"157fadd2", "dfaddb83"}:
                row["privateSurface"] = group[0].key
            owners.append(row)
        reviewed[sid] = {
            "projection": "container_occurrences" if prefix == "4ae247ca" else "shipment",
            "declarations": owners,
            "sourceContractScope": "Individually inspected package count/private-clause owners; "
            "retain all headers and inner/support relationships. Quantity-only slots gain "
            "an explicit disjoint-group list; no absent per-container split is invented.",
        }
        if prefix == "b8544221":
            reviewed[sid]["planning"] = "shared_units"
    if len(reviewed) != len(REVIEWED):
        raise ValueError("reviewed source list is not fully present")
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    prefixes = list(REVIEWED)
    for batch, start in enumerate(range(0, len(prefixes), 20), 24):
        compile_batch(
            path,
            root / f"batch-{batch:02}",
            prefixes[start : start + 20],
            2,
            group_counts=[1, 2, 3],
            context=context,
            reviewed=reviewed,
        )
    print(json.dumps({"reviewedSources": len(reviewed), "batches": [24, 25]}))


if __name__ == "__main__":
    main()
