"""Render certified complete declarations as ordered public package lists.

The original scalar count/type representation is an internal rendering view of
the aggregate, never an extra public package row. Complete lists are composed
before byte rendering and description/coordinate projection, not patched into
finished text.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from document_ocr.synthesis.curated import digest, flat
from document_ocr.synthesis.curated_casing import case_owned_text
from document_ocr.synthesis.curated_templates import RenderRegion
from document_ocr.synthesis.declared_packages import DeclaredPackages
from document_ocr.synthesis.template_compiler.descendant import _number_to_words

PREFIX = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages"


def project_geography_contract(original, projected, contract: dict) -> dict:
    """Re-pin an authorized locality/country collapse, preserving all cargo owners."""
    from document_ocr.synthesis.independent_geography import validate_locality_target_projection

    if (
        contract["sourceSha256"] != digest(original.source.encode())
        or contract["sourceTargetSha256"] != digest(original.target)
        or original.source != projected.source
    ):
        raise ValueError("geography projection lacks the original pinned package contract")
    changed = validate_locality_target_projection(original, projected)
    old = {(r.key, r.occurrence): r for r in original.regions}
    new = {(r.key, r.occurrence): r for r in projected.regions}
    if any(
        old[tuple(key)] != new.get(tuple(key))
        for owner in contract["declarations"]
        for key in owner["owners"]
    ):
        raise ValueError("geography projection changed a package declaration owner")
    return {
        **contract,
        "sourceTargetSha256": digest(projected.target),
        "geographyProjection": {
            "originalContractSha256": digest(contract),
            "omittedPaths": sorted(changed),
        },
    }


def package_rows(target: dict) -> list[dict]:
    return target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"]


def aggregate_target(target: dict, category: str) -> dict:
    result = deepcopy(target)
    rows = package_rows(result)
    if not rows or any(set(r) != {"packageQuantity", "typeCategory"} for r in rows):
        raise ValueError("declared-group rendering requires exact count/category rows")
    result["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"] = [
        {"packageQuantity": sum(r["packageQuantity"] for r in rows), "typeCategory": category}
    ]
    return result


def public_rows(ledger: DeclaredPackages, projection: str) -> list[dict]:
    if projection == "shipment":
        return ledger.public_rows()
    if projection != "container_occurrences":
        raise ValueError("unknown declared-group public projection")
    return [
        {"packageQuantity": n, "typeCategory": group.category}
        for row in ledger.matrix
        for n, group in zip(row, ledger.groups, strict=True)
        if n
    ]


@dataclass(frozen=True)
class PackageListRendering:
    source_hash: str
    target_rows: list[dict]
    aggregate_category: str
    regions: tuple[tuple[RenderRegion, str], ...]
    replaced: frozenset[tuple[str, int]]
    evidence: dict[str, Any]

    def validate_witnesses(self, vocabulary) -> None:
        """Validate public rows, never require printing the internal aggregate type."""
        if vocabulary is None:
            raise ValueError("declared package witness validation requires a vocabulary")
        for index, row in enumerate(self.target_rows):
            path = f"{PREFIX}[{index}].typeCategory"
            if not any(
                path in region.target_paths and vocabulary.supports(text, row["typeCategory"])
                for region, text in self.regions
            ):
                raise ValueError("public package row lacks its own printed type witness")

    def view(self, blueprint, target: dict) -> dict:
        if (
            digest(blueprint.source.encode()) != self.source_hash
            or package_rows(target) != self.target_rows
        ):
            raise ValueError("package list rendering differs from its source or public groups")
        return aggregate_target(target, self.aggregate_category)

    def compose(
        self, existing: list[tuple[RenderRegion, str]], *, casing: str
    ) -> list[tuple[RenderRegion, str]]:
        actual = {(r.key, r.occurrence) for r, _ in existing}
        if not self.replaced <= actual:
            raise ValueError("package list consumed owner is absent from rendering")
        result = [(r, text) for r, text in existing if (r.key, r.occurrence) not in self.replaced]
        result.extend(
            (r, case_owned_text(text, r.target_paths, casing)) for r, text in self.regions
        )
        result.sort(key=lambda item: item[0].start)
        if any(a.end > b.start for (a, _), (b, _) in pairwise(result)):
            raise ValueError("package list crosses an unrelated source owner")
        return result

    def project(self, blueprint, target: dict, proof: dict) -> None:
        target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"] = deepcopy(
            self.target_rows
        )
        before, after = flat(blueprint.target), flat(target)
        paths = {
            p
            for p in before.keys() | after.keys()
            if p.startswith(PREFIX + "[") and before.get(p) != after.get(p)
        }
        for field in ("changedTargetPaths", "coveredTargetPaths"):
            proof[field] = sorted(
                {p for p in proof[field] if not p.startswith(PREFIX + "[")} | paths
            )
        proof["declaredPackageLists"] = deepcopy(self.evidence)

    def alias_receipt(self, receipt: dict) -> dict:
        result = deepcopy(receipt)
        result["slots"] = [
            s for s in result["slots"] if (s["key"], s["occurrence"]) not in self.replaced
        ]
        result["nonScalarOwners"] = [
            s for s in result["nonScalarOwners"] if (s["key"], s["occurrence"]) not in self.replaced
        ] + [
            {"key": r.key, "occurrence": r.occurrence, "mode": "compound"} for r, _ in self.regions
        ]
        result["declaredPackageLists"] = deepcopy(self.evidence)
        return result


def prepare_lists(
    blueprint,
    target,
    ledger: DeclaredPackages,
    contract: dict,
    presentations,
    *,
    seed,
    variant,
    private_packing=None,
) -> PackageListRendering:
    """A reviewed contract names exact source spans and accounting scopes."""
    if contract["sourceSha256"] != digest(blueprint.source.encode()) or contract[
        "sourceTargetSha256"
    ] != digest(blueprint.target):
        raise ValueError("declared-group source contract is stale")
    projection = contract["projection"]
    rows = public_rows(ledger, projection)
    if package_rows(target) != rows:
        raise ValueError("public package rows differ from the private ledger projection")
    vocabulary = presentations.vocabulary
    families = {vocabulary.entries[r["typeCategory"]]["family"] for r in rows}
    aggregate = next(iter(families)) if len(families) == 1 else "PACKAGE_PACKAGE"
    source_regions = {(r.key, r.occurrence): r for r in blueprint.regions}
    raw = blueprint.source.encode()
    made, replaced, receipts, covered = [], set(), [], set()
    local_indices = []
    cursor = 0
    for row in ledger.matrix:
        n = sum(q > 0 for q in row)
        local_indices.append(list(range(cursor, cursor + n)))
        cursor += n
    for i, owner in enumerate(contract["declarations"]):
        keys = [tuple(k) for k in owner["owners"]]
        if not keys or len(set(keys)) != len(keys) or replaced.intersection(keys):
            raise ValueError("declaration owners must be nonempty and disjoint")
        originals = [source_regions[k] for k in keys]
        start, end = min(r.start for r in originals), max(r.end for r in originals)
        if (start, end, raw[start:end].decode()) != (owner["start"], owner["end"], owner["source"]):
            raise ValueError("complete package declaration span changed")
        inside = [r for r in blueprint.regions if r.start < end and start < r.end]
        if {(r.key, r.occurrence) for r in inside} != set(keys):
            raise ValueError("complete declaration includes an unowned intervening region")
        # Audited complete phrases may include labels/separators. Their exact
        # bytes are pinned above; automatic compilation only merges whitespace.
        scope = owner["scope"]
        if scope == "shipment":
            if projection != "shipment":
                raise ValueError("container-local public groups cannot gain a global identity")
            indices = list(range(len(rows)))
            selected = rows
        elif type(scope) is int and 0 <= scope < len(ledger.matrix):
            selected = [
                {"packageQuantity": q, "typeCategory": g.category}
                for q, g in zip(ledger.matrix[scope], ledger.groups, strict=True)
                if q
            ]
            indices = local_indices[scope] if projection == "container_occurrences" else []
        else:
            raise ValueError("invalid package declaration accounting scope")
        terms, forms = [], []
        count_styles = {r.presentation for r in originals if r.kind == "count"}
        source_format = (
            "words_digits"
            if {"words", "number"} <= count_styles
            else "words"
            if count_styles == {"words"}
            else "digits"
        )
        for j, row in enumerate(selected):
            category = row["typeCategory"]
            protected = contract.get("protectedForms", {}).get(category)
            if protected:
                word, choice = protected[0], {"style": "protected"}
            else:
                styles = (
                    ("full", "optional")
                    if vocabulary.entries[category]["family"] != category
                    else ("full", "short", "optional")
                )
                word, choice = vocabulary.choose(
                    category,
                    styles,
                    presentations.policy.weights(),
                    f"declared-groups:{seed}:{blueprint.document_id}:{variant}:{i}:{j}",
                )
                if not vocabulary.supports(word, category):
                    raise ValueError("declared package group lost its specific witness")
            count = row["packageQuantity"]
            quantity_format = owner.get("quantityFormat", source_format)
            if quantity_format == "words":
                quantity = _number_to_words(count)
            elif quantity_format == "words_digits":
                quantity = f"{_number_to_words(count)} ({count})"
            elif quantity_format == "digits":
                quantity = str(count)
            elif quantity_format == "decimal2":
                quantity = f"{count}.00"
            else:
                raise ValueError("unknown declared package quantity presentation")
            terms.append(f"{quantity} {word}")
            forms.append({**row, "text": word, **choice})
        text = " + ".join(terms)
        if owner.get("privateSurface") is not None:
            key = owner["privateSurface"]
            if keys != [(key, 0)] or private_packing is None:
                raise ValueError("private declaration must retain its complete numeric owner")
            values = private_packing["surfaces"][key]
            if len(values) != 1 or not values[0].strip():
                raise ValueError("private declaration has no complete generated clause")
            text = values[0]
            if any(not vocabulary.supports(text, row["typeCategory"]) for row in selected):
                raise ValueError("private declaration lost a public category witness")
        if owner.get("lineBreak"):
            text = text.replace(" + ", "\n+ ")
        paths = tuple(
            f"{PREFIX}[{index}].{field}"
            for index in indices
            for field in ("packageQuantity", "typeCategory")
        )
        region = RenderRegion(
            f"declared-packages:{i}", 0, start, end, raw[start:end].decode(), paths, "owned_text"
        )
        made.append((region, text))
        replaced.update(keys)
        covered.update(indices)
        receipts.append(
            {
                "key": region.key,
                "scope": scope,
                "indices": indices,
                "forms": forms,
                "start": start,
                "end": end,
                "source": region.source,
                "text": text,
                **(
                    {"privateSurface": owner["privateSurface"]} if "privateSurface" in owner else {}
                ),
            }
        )
    if covered != set(range(len(rows))):
        raise ValueError("not every public package group has a printed declaration")
    evidence = {
        "contractSha256": digest(contract),
        "ledgerSha256": digest(ledger.receipt()),
        "projection": projection,
        "rows": rows,
        "declarations": receipts,
    }
    return PackageListRendering(
        digest(raw), rows, aggregate, tuple(made), frozenset(replaced), evidence
    )
