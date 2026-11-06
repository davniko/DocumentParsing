"""Authorized identifier-only long inventories, not donor-free admission.

Both reviewed sources lack published production contracts. These bounded
contracts own only identifiers; cargo, anonymized parties and all other text
remain fixed. Three variants per source replace six older synthetic records.
"""

import copy
import json
import re
import time
from collections import Counter
from pathlib import Path

from finalize_r28_expansion import feature_function
from finalize_r29_equal_loads import features, task
from refresh_r29_equal_loads import OUT, SOURCE, load

from document_ocr.atomic import atomic_write_json
from document_ocr.synthesis.curated import SourceContract, assign, digest, flat, locate
from document_ocr.synthesis.curated_reflow_policy import FontMetrics
from document_ocr.synthesis.generators import (
    DeterministicStream,
    generate_from_surface_pattern,
    generate_unique_container_numbers,
    surface_pattern,
    validate_container_number,
)
from document_ocr.synthesis.raw_text_template import (
    build_template_slot,
    compile_raw_text_template,
    render_compiled_template,
)

SOURCE_PREFIXES = ("doc_48b2d8a5", "doc_01c3694f")


def compile_identifiers(row, contract):
    slots, keys = [], {}
    for variable in contract.variables:
        for occurrence in variable.occurrences:
            start, end = locate(row["joinedRawText"], occurrence)
            key = f"slot_{len(slots):04d}"
            slots.append(
                build_template_slot(
                    slot_id=key,
                    byte_start=start,
                    byte_end=end,
                    source_text=occurrence.text,
                    target_paths=[
                        t.path for t in contract.targets if t.expression == "{" + variable.key + "}"
                    ],
                    semantic_role=variable.meaning,
                    evidence_origin="accepted_label_evidence",
                    render_policy="opaque_identifier",
                )
            )
            keys[key] = variable.key
    template = compile_raw_text_template(
        document_id=row["documentId"], source=row["joinedRawText"].encode(), slots=slots
    )
    return template, keys


def render_identifiers(row, contract, values):
    assert set(values) == {v.key for v in contract.variables}
    template, keys = compile_identifiers(row, contract)
    for variable in contract.variables:
        assert surface_pattern(values[variable.key]) == surface_pattern(variable.value)
        if variable.kind == "container":
            assert validate_container_number(values[variable.key])
    payload, proof = render_compiled_template(
        source=row["joinedRawText"].encode(),
        template=template,
        bindings={slot: values[key] for slot, key in keys.items()},
    )
    target = copy.deepcopy(row["target"])
    for binding in contract.targets:
        assign(target, binding.path, values[binding.expression[1:-1]])
    return payload.decode(), target, proof.model_dump(mode="json")


def identifier_contract(row):
    leaves = flat(row["target"])
    owners = {}
    for path, value in leaves.items():
        if (
            path == "billOfLadingNumber"
            or path.endswith(".equipmentIdentifier")
            or re.search(r"\.sealNumbers\[\d+\]$", path)
        ):
            owners.setdefault(value, []).append(path)
    variables, targets = [], []
    for index, (value, paths) in enumerate(owners.items()):
        key = f"identifier_{index:03d}"
        raw = row["joinedRawText"]
        # A seal may be a suffix of the B/L number. Own standalone tokens,
        # while occurrence indices still refer to every literal substring.
        matches = list(
            re.finditer(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", raw)
        )
        if not matches:
            raise ValueError("identifier is not present in source: " + value)
        variables.append(
            dict(
                key=key,
                value=value,
                kind="container"
                if any(p.endswith(".equipmentIdentifier") for p in paths)
                else "identifier",
                meaning="Shared shipment identifier, every printed occurrence and target link.",
                required_literals=[],
                occurrences=[
                    dict(
                        text=value,
                        occurrence=raw[: m.start()].count(value) + 1,
                        presentation="text",
                    )
                    for m in matches
                ],
            )
        )
        targets.extend(dict(path=p, expression="{" + key + "}") for p in paths)
    return SourceContract.model_validate(
        dict(
            variables=variables,
            targets=targets,
            fixed_context=(
                "SOURCE-PRESERVING ONLY, NOT DONOR-FREE. Keep cargo, quantities, weights, "
                "types, anonymized parties, other private references, routes, dates, "
                "source omissions and all coordinates unchanged. Only B/L, container "
                "and seal identifiers vary. No production admission is asserted."
            ),
        )
    )


def positioned_overlay(before, raw):
    output = []
    for old, new, located in zip(
        before["joinedRawText"].splitlines(keepends=True),
        raw.splitlines(keepends=True),
        before["positionedText"].splitlines(keepends=True),
        strict=True,
    ):
        old, new = old.rstrip("\r\n"), new.rstrip("\r\n")
        if not located.startswith(old):
            raise ValueError("positioned text does not match raw line")
        suffix = located[len(old) :]
        if suffix.strip() and not re.fullmatch(r" \|\|(?: \d+,\d+)?\s*", suffix):
            raise ValueError("unrecognized coordinate suffix")
        output.append(new + suffix)
    return "".join(output)


def generate(source, contract, *, variant, prefix, excluded, metrics):
    stream = DeterministicStream(
        2026110229, "long-source-identifiers", f"{source['documentId']}:{variant}"
    )
    container_vars = [v for v in contract.variables if v.kind == "container"]
    ids = generate_unique_container_numbers(
        owner_by_identity={v.value: prefix for v in container_vars},
        stream=stream,
        excluded=excluded,
    )
    values = {}
    for variable in contract.variables:
        if variable.kind == "container":
            value = ids[variable.value]
            assert validate_container_number(value)
        else:
            for attempt in range(1000):
                value = generate_from_surface_pattern(
                    pattern=surface_pattern(variable.value),
                    stream=stream.derive(f"{variable.key}:{attempt}"),
                    excluded=excluded | set(values.values()),
                    additional_excluded=variable.value,
                )
                if metrics.length(value) <= metrics.length(variable.value):
                    break
            else:
                raise ValueError("could not produce a nonexpanding identifier")
        assert metrics.length(value) <= metrics.length(variable.value)
        values[variable.key] = value
    raw, target, proof = render_identifiers(source, contract, values)
    assert raw != source["joinedRawText"]
    located = positioned_overlay(source, raw)
    after = copy.deepcopy(source)
    after.update(
        joinedRawText=raw,
        joinedRawTextSha256=digest(raw.encode()),
        positionedText=located,
        positionedTextSha256=digest(located.encode()),
        target=target,
    )
    return after, dict(values=values, proof=proof, sourceSha256=digest(source), prefix=prefix)


def main():
    started = time.perf_counter()
    authority, flags = task(), feature_function()
    font = load("configs/synthesis/mpci_bl_curated_v7_multirow.json")["positions"]["reflow"]
    metrics = FontMetrics(Path(font["font_path"]), font["font_sha256"])
    real = list(map(json.loads, (SOURCE / "real-train.jsonl").open()))
    synthetic = list(map(json.loads, (SOURCE / "synthetic-train.jsonl").open()))
    sources = [next(r for r in real if r["documentId"].startswith(p)) for p in SOURCE_PREFIXES]
    used = set(load(OUT / "selection.json")["documents"])
    used.update(r["id"] for r in load(OUT / "prefix-summary.json")["rows"])
    used.update(r["documentId"] for r in load(OUT / "legacy-summary.json")["rows"])
    old_ids = {
        r["id"]
        for r in load("docs/analysis/paddle-migration-scope-20261102/cargo-refresh-candidates.json")
        if r["old"]
    }
    excluded = {
        str(value)
        for name in ("synthetic-train.jsonl", "real-train.jsonl", "validation.jsonl")
        for row in map(json.loads, (SOURCE / name).open())
        for value in flat(row["target"]).values()
        if isinstance(value, str)
    }
    rows = []
    for source in sources:
        contract = identifier_contract(source)
        template, _ = compile_identifiers(source, contract)
        sid = source["documentId"]
        atomic_write_json(
            OUT / "long-source-contracts" / f"{sid}.json",
            dict(
                sourceSha256=digest(source),
                contract=contract.model_dump(),
                compiledSlots=len(template.slots),
                mode="SOURCE_PRESERVING_ONLY_NOT_DONOR_FREE",
                fontSha256=font["font_sha256"],
            ),
        )
        prefix = min(
            {v.value[:4] for v in contract.variables if v.kind == "container"},
            key=lambda s: (metrics.length(s), s),
        )
        sf = flags(source["target"])
        for variant in range(1, 4):
            candidates = []
            for old in synthetic:
                did = old["documentId"]
                if did in used or did not in old_ids:
                    continue
                f, ff = features(old), flags(old["target"])
                if not (
                    f["complete"]
                    and not f["equal"]
                    and not f["mixed"]
                    and f["containers"] < 10
                    and f["packageRows"] == 1
                ):
                    continue
                if any(
                    bool(ff.get(k)) != bool(sf.get(k))
                    for k in (
                        "dg",
                        "thermal",
                        "forwarding",
                        "delivery",
                        "masterBL",
                        "vesselIMO",
                        "reefer",
                        "tank",
                        "openTop",
                        "platform",
                    )
                ):
                    continue
                if bool(old["target"].get("route", {}).get("transshipmentPort")) != bool(
                    source["target"].get("route", {}).get("transshipmentPort")
                ):
                    continue
                if (
                    old["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0][
                        "typeCategory"
                    ]
                    != "PACKAGE_PACKAGE"
                ):
                    continue
                candidates.append(old)
            if not candidates:
                raise ValueError("no compatible older replacement without coverage loss")
            before = min(candidates, key=lambda r: digest([sid, variant, r["documentId"]]))
            after, receipt = generate(
                source, contract, variant=variant, prefix=prefix, excluded=excluded, metrics=metrics
            )
            assert authority.canonicalize(after["target"]) == after["target"]
            assert features(after)["equal"] and features(after)["containers"] >= 10
            # Refresh membership identity is stable; its new source parent is explicit.
            after["documentId"] = before["documentId"]
            after["sourceDocumentId"] = sid
            assert render_identifiers(source, contract, receipt["values"])[:2] == (
                after["joinedRawText"],
                after["target"],
            )
            excluded.update(receipt["values"].values())
            used.add(before["documentId"])
            receipt.update(
                variant=variant,
                contractSha256=digest(contract.model_dump()),
                coordinatePolicy="unchanged source anchors; every changed footprint nonexpanding",
            )
            path = OUT / "staged-long" / f"{before['documentId']}.json"
            atomic_write_json(
                path, dict(before=before, source=source, after=after, evidence=receipt)
            )
            rows.append(
                dict(
                    id=before["documentId"],
                    parent=sid,
                    variant=variant,
                    containers=features(after)["containers"],
                    prefix=prefix,
                    stagedSha256=digest(load(path)),
                )
            )
    atomic_write_json(
        OUT / "long-source-summary.json",
        dict(
            status="VALIDATED_SOURCE_PRESERVING_NOT_DONOR_FREE",
            rows=rows,
            records=len(rows),
            parents=dict(Counter(r["parent"] for r in rows)),
            admissionFinding=(
                "Neither source has a published historical or current synthesis contract; "
                "01c3694f historical compiler result rejected with no independently confirmed "
                "template. Full admission requires new lexical/numeric/source-only ownership "
                "and geometry certification, not merely rebasing. Authorized identifier-only "
                "option used."
            ),
            seconds=time.perf_counter() - started,
        ),
    )
    print(json.dumps(dict(records=len(rows), seconds=time.perf_counter() - started)))


if __name__ == "__main__":
    main()
