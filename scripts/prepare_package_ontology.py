"""Pin manually reviewed crosswalk shards; do not infer properties or edit inputs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "configs/ontology/package-properties-v1"
ATTRIBUTE_DEFINITIONS = {
    "CAGING": "Explicit presence/absence of a cage; not absence of every package or support.",
    "CLOSURE": "Explicit opening/head/lid form; omission means neither open nor closed.",
    "COATING": "Explicit coating state of the stated body, separate from lining.",
    "COLOUR": "Explicit colour in a named package definition.",
    "COMMODITY_MATERIAL": "Substance of cargo form, not the primary packaging-body material.",
    "COMPRESSION": "Explicit compressed/non-compressed bale presentation.",
    "CONSTRUCTION": "Explicit fabrication/structural variant; no usual construction is inferred.",
    "CONTENTS_ARRANGEMENT": "Expressly defined arrangement of contents, not a sampled quantity.",
    "CONTENTS_FORM": "Stated contents or service form, not package-body substance.",
    "CONTENTS_SHAPE": "Shape of stated contents, separate from the enclosing body's shape.",
    "COUNT_FORM": "Named counting form; never a deduction that shipment quantity is one.",
    "COVER": "Explicit presence/absence of a cover, not an assumption about an unstated lid.",
    "DIMENSION_CLASS": "Named dimensional class without invented numeric dimensions.",
    "EXCLUDED_MATERIAL": "Explicit excluded material; not selection of an alternative.",
    "FOOTPRINT": "Named relative footprint; no unprinted dimensions or units are inferred.",
    "GRIP_HOLES": "Explicit grip-hole feature/purpose.",
    "GROUPING": "Named bundle/bunch/truss grouping alternatives; no unprinted member count.",
    "HANDLE": "Explicit handle feature, independent of body material.",
    "LAYER": "Named package-layer designation; the document still determines its accounting unit.",
    "LAYERS": "Explicit number-of-layers designation of a tray/package configuration.",
    "LINING": "Explicit presence/absence of lining; liner material remains component-owned.",
    "NECK": "Expressly defined neck form, not an inferred closure.",
    "NOZZLE": "Explicit nozzle feature; this alone does not establish a closing cap.",
    "PACKAGING_PROCESS": "Explicit wrapping/treatment process, independent of package family.",
    "PACKING_STATE": "Explicit packing state; not inference of a particular enclosure.",
    "PARTICLE_FORM": "Expressly defined bulk particle/form distinction.",
    "POLYMER": "Explicit polymer refinement of a broader plastic body material.",
    "POOL": "Named pool; no inferred country, owner, material or dimensions.",
    "PRESSURIZATION": "Explicit package pressure class; not measured equipment pressure.",
    "PROFILE": "Named relative package profile, without invented dimensions.",
    "PROPULSION": "Explicit propulsion; not fuel, DG status or document carrying role.",
    "PROTECTION": "Explicit protected/non-protected definition; omission is not a negative.",
    "PURPOSE": "Purpose expressly required by a definition, excluding merely typical use examples.",
    "RETURNABILITY": "Explicit one-way/returnable use-cycle designation.",
    "RIGIDITY": "Explicit rigid/flexible distinction for the stated body or component.",
    "RIM": "Explicit rim feature; dimensional facts are represented separately.",
    "SHAPE": "Explicit shape of the named body/component, separate from cargo contents shape.",
    "SIDES": "Explicit side-count designation of a handling form.",
    "SIFT_PROOF": "Explicit sift-proof statement; omission/ordinary does not imply its negation.",
    "SIZE": "Named relative size class; no numeric dimensions are implied.",
    "SPOUT": "Explicit spout feature, not an unstated lid/closure.",
    "STACKABILITY": "Expressly named stacking property.",
    "STATUS": "Administrative meaning, not a physical family or package-material fact.",
    "STRUCTURAL_EQUIPMENT": "Explicit supporting equipment/freestanding construction of an IBC.",
    "TEMPERATURE_PRESSURE_CONDITION": "Joint condition; no invented thresholds or setpoints.",
    "THERMAL_PROPERTY": "Package thermal construction, not an equipment setpoint.",
    "TRADE_FORM": "Named trade variant not safely discarded or treated as a universal synonym.",
    "TRANSPARENCY": "Definition's explicit transparency; not implied by a generic wrapping noun.",
    "TRANSPORT_EQUIPMENT": "Definition qualification; actual document role still needs ownership.",
    "WATER_RESISTANCE": "Explicit water resistance; omission does not imply the opposite.",
    "WHEELS": "Explicit wheeled construction, not an inference from every movable support.",
    "WOOD_TYPE": "Explicit natural wood, distinct from unspecified wood/manufactured boards.",
}


def pin(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    files = sorted(DIRECTORY.glob("entries-*.json"))
    assert len(files) == 4
    rows = [row for path in files for row in json.loads(path.read_text())]
    assert len(rows) == len({row["code"] for row in rows}) == 405
    values = defaultdict(set)
    for row in rows:
        for obj in (row, *row.get("components", [])):
            for key, value in obj.get("attributes", {}).items():
                values[key].add(value)
    assert values.keys() == ATTRIBUTE_DEFINITIONS.keys(), (
        values.keys() - ATTRIBUTE_DEFINITIONS.keys(),
        ATTRIBUTE_DEFINITIONS.keys() - values.keys(),
    )
    manifest = {
        "schemaVersion": 1,
        "status": "internal_reviewed_reference_not_runtime_enabled",
        "reviewedAt": "2026-10-25",
        "sourceRegistry": pin(
            ROOT
            / "artifacts/mpci-ai-schema/categorical-registry-followup"
            / "package-category-registry.json",
            ROOT,
        ),
        "officialEntryCount": 405,
        "entryFiles": [pin(path, DIRECTORY) for path in files],
        "semanticExtensions": pin(DIRECTORY / "semantic-extensions.json", DIRECTORY),
        "requiredActiveTask": pin(
            ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/task-constraints.json", ROOT
        ),
        "requiredAdditionalFamilies": ["BARREL", "JERRICAN", "BUCKET", "TUB", "KIT"],
        "primarySources": [
            {
                "id": "unece_package_names_20261023",
                "url": "https://vocabulary.uncefact.org/PackageTypeCodeList",
                "sha256": "1357f26aa01fdf3dd2be951135397bba97dd32483835de9f86792cce6c883d09",
            },
            {
                "id": "oasis_ubl21_rec21_9e",
                "url": "https://docs.oasis-open.org/ubl/os-UBL-2.1/cl/gc/default/PackagingTypeCode-2.1.gc",
                "sha256": "f318a864d92ca35ee1b94e2a25fb5ad646f802c7d792e452cb950fea15785fa1",
            },
            {
                "id": "un_adr2025",
                "url": "https://www.mindop.sk/fileadmin/documents/doprava/nebezpecnetovary/cesta/ADR_2025_Vol._II._EN.pdf",
                "sha256": "702d0ae5737d3bd228e0b332352deda626018ac9841339c0eca67e4209953f4b",
            },
            {
                "id": "gs1_v116",
                "url": "https://raw.githubusercontent.com/gs1/WebVoc/master/v1.16/gs1Voc.jsonld",
                "sha256": "5b417c158cca7e55766dc5e35a1fa480273f036720b6c7382d61ea6939a7ee66",
            },
        ],
        "attributeDefinitions": dict(sorted(ATTRIBUTE_DEFINITIONS.items())),
        "attributeVocabulary": {key: sorted(vs) for key, vs in sorted(values.items())},
        "notes": [
            (
                "This reference preserves accepted code identities and reviewed definition facts; "
                "it is not a public extraction-schema migration or alias registry."
            ),
            (
                "Supplemental OASIS definitions are UNECE revision 9e adopted by UBL 2.1, not "
                "a claim that this is the newest list. Saved names remain authoritative for "
                "identity; edition differences are retained in entry notes."
            ),
            (
                "Absent attributes are valid unasserted properties, never guessed negatives or "
                "requirements to choose a more-specific child."
            ),
            (
                "Formal code facts and ordinary-language evidence are separate. A generic name "
                "must not automatically inherit every formal definition fact, especially those "
                "from a supplemental definition source."
            ),
            (
                "Distinct codes can have equivalent properties. This reference and family/material "
                "projection do not invent a unique inverse submission code."
            ),
            (
                "Measurements belong to definitions, not shipment equipment settings; unknown "
                "dimension units and pressure comparators remain unknown."
            ),
            (
                "All 48 active meanings and five additional pilot families are required. Full "
                "code coverage does not enable all sampling categories or change targets."
            ),
        ],
    }
    output = DIRECTORY / "manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "official": len(rows),
                "attributes": len(values),
                "manifest": str(output.relative_to(ROOT)),
            }
        )
    )


if __name__ == "__main__":
    main()
