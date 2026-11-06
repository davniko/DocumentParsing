"""Replay explicitly approved identifier-only cohorts, not commodity synthesis.

These receipts retain a certified Paddle source, including its OCR absences and
accounting conflicts. They must never be used as an arithmetic fallback for a
newly sampled shipment. Every non-identifier target and input span is conserved.
"""

from __future__ import annotations

import copy
import json
import re

from document_ocr.spatial_inputs.paddle import RegionSpan, pages_from_record
from document_ocr.synthesis.curated import SourceContract, digest
from document_ocr.synthesis.paddle_bindings import PaddleReplacement
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog, read_pin, verified_path


def identifier_mapping(contract, values, projections):
    variables = {v["key"]: v for v in contract["variables"]}
    if set(values) != set(variables):
        raise ValueError("source-preserving values differ from contract membership")
    if set(projections) - set(variables):
        raise ValueError("unknown native identifier projection")
    result = {}
    for key, variable in variables.items():
        old, new = variable["value"], values[key]
        projection = projections.get(key)
        if projection is not None:
            if projection["original"] != old:
                raise ValueError("native identifier projection belongs to another owner")
            native = projection["native"]
            mask = {int(i): ch for i, ch in projection["retainedOcrMask"].items()}
            if any(not 0 <= i < len(new) or len(ch) != 1 for i, ch in mask.items()):
                raise ValueError("invalid retained OCR mask")
            new = "".join(mask.get(i, ch) for i, ch in enumerate(new))
            old = native
        if not all(isinstance(v, str) and v.isascii() and v.isalnum() for v in (old, new)):
            raise ValueError("identifier-only receipt contains non-identifier text")
        if len(old) != len(new) or old == new or old in result:
            raise ValueError("identifier replacement is duplicate, unchanged or changes format")
        result[old] = new
    if len(set(result.values())) != len(result) or set(result) & set(result.values()):
        raise ValueError("identifier replacement is not a disjoint bijection")
    return result


def identifier_proof(source, mapping):
    proof, counts = [], dict.fromkeys(mapping, 0)
    for page in pages_from_record(source):
        for region in page.regions:
            for match in re.finditer(r"[A-Za-z0-9]+", region.text):
                token, queue, complete = match.group(), [(0, [])], []
                while queue:
                    pos, parts = queue.pop()
                    if pos == len(token):
                        complete.append(parts)
                        continue
                    for before in mapping:
                        if token.startswith(before, pos):
                            queue.append((pos + len(before), parts + [(pos, before)]))
                if not complete:
                    if any(before in token for before in mapping):
                        raise ValueError("identifier has an unreviewed joined-text occurrence")
                    continue
                if len(complete) != 1:
                    raise ValueError("ambiguous concatenated identifier ownership")
                for offset, before in complete[0]:
                    start = match.start() + offset
                    proof.append(PaddleReplacement(
                        f"source-identifier:{len(proof)}",
                        (RegionSpan(page.index, region.index, start, start + len(before), before),),
                        mapping[before],
                    ))
                    counts[before] += 1
    if not all(counts.values()):
        raise ValueError("native source lacks a contracted identifier occurrence")
    return tuple(proof), counts


def identifier_target(target, mapping):
    def visit(value, path=""):
        if isinstance(value, dict):
            return {k: visit(v, path + "/" + k) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v, path + "/" + str(i)) for i, v in enumerate(value)]
        if isinstance(value, str) and value in mapping:
            allowed = path == "/billOfLadingNumber" or re.fullmatch(
                r"/containerInformation/\d+/(equipmentIdentifier|sealNumbers/\d+)", path
            ) or re.fullmatch(
                r"/goodsItemDetails/\d+/splitGoodsPlacement/\d+/equipmentIdentifier", path
            )
            if not allowed:
                raise ValueError("identifier replacement would change a non-identifier target")
            return mapping[value]
        return value
    return visit(target)


def accounting_signature(target):
    """Expose, rather than repair, the source's incomplete public inventory."""
    equipment = [c["equipmentIdentifier"] for c in target.get("containerInformation", [])]
    if len(equipment) != len(set(equipment)):
        raise ValueError("duplicate source equipment")
    signature = []
    for goods in target.get("goodsItemDetails", []):
        allocations = goods.get("splitGoodsPlacement", [])
        ids = [a["equipmentIdentifier"] for a in allocations]
        if len(ids) != len(set(ids)) or not set(ids) <= set(equipment):
            raise ValueError("invalid source allocation identity")
        signature.append({
            "equipmentCount": len(equipment),
            "allocationQuantities": [a.get("packageQuantity") for a in allocations],
            "packages": copy.deepcopy(goods.get("numberAndTypeOfPackages", [])),
        })
    return signature


class R30Replay(PaddleTemplateCatalog):
    """Ordinary catalog replay plus a finite, explicit migration receipt set."""

    def __init__(self, root, catalog_pin, source_preserving=()):
        super().__init__(root, catalog_pin)
        self.source_preserving = {}
        self._source_paths = {}
        for pin in source_preserving:
            receipt = read_pin(self.root, pin)
            identity = receipt["documentId"]
            if receipt["status"] != "REVIEWED_SOURCE_PRESERVING_REBASE" or identity in self.source_preserving:
                raise ValueError("source-preserving registry is unreviewed or duplicated")
            path = verified_path(self.root, pin).absolute()
            self.source_preserving[identity] = path
            self._source_paths[path] = (pin, receipt)

    def wording_specification(self, candidate_path):
        if candidate_path.absolute() in self._source_paths:
            return None
        return super().wording_specification(candidate_path)

    def render(self, candidate_path, *, current=None, wording_receipt=None):
        entry = self._source_paths.get(candidate_path.absolute())
        if entry is None:
            return super().render(candidate_path, current=current, wording_receipt=wording_receipt)
        from stage_r30_independent_repairs import render_proof

        pin, receipt = entry
        if read_pin(self.root, pin) != receipt:
            raise ValueError("source-preserving receipt changed during replay")
        if current is None or digest(current) != receipt["currentRecordSha256"] or wording_receipt is not None:
            raise ValueError("source-preserving replay has different current data or unsolicited wording")
        stage = read_pin(self.root, receipt["previousStage"])
        contract = read_pin(self.root, receipt["contract"])
        if digest(stage["source"]) != contract["sourceSha256"] or contract["sourceSha256"] != stage["evidence"]["sourceSha256"]:
            raise ValueError("source-preserving original source differs")
        if digest(SourceContract.model_validate(contract["contract"]).model_dump()) != stage["evidence"]["contractSha256"]:
            raise ValueError("source-preserving contract differs")
        for key, value in stage["after"].items():
            if key not in ("joinedRawTextSha256", "positionedTextSha256") and current[key] != value:
                raise ValueError("source-preserving historical shipment differs from current data")
        source = read_pin(self.root, receipt["nativeSource"])
        certificate = read_pin(self.root, receipt["sourceCertificate"])
        source_rows = verified_path(self.root, certificate["records"]["real-train"])
        matches = [json.loads(s) for s in source_rows.open() if json.loads(s)["documentId"] == receipt["sourceDocumentId"]]
        if matches != [source] or source["documentId"] != current["sourceDocumentId"]:
            raise ValueError("native source is not the exact certified real record")
        mapping = identifier_mapping(contract["contract"], stage["evidence"]["values"], receipt["nativeIdentifierProjections"])
        proof, occurrences = identifier_proof(source, mapping)
        target = identifier_target(source["target"], mapping)
        signature = accounting_signature(source["target"])
        if signature != receipt["preservedSourceAccounting"] or accounting_signature(target) != signature:
            raise ValueError("source accounting changed outside identifier-only scope")
        if identifier_target(target, {v: k for k, v in mapping.items()}) != source["target"]:
            raise ValueError("source-preserving target does not invert exactly")
        self.validator.validate(target)
        rendered = render_proof(source, proof, {"physicalGroups": [], "watermarkRegions": []}, self.engine, current["documentId"])
        rendered.update(documentId=current["documentId"], sourceDocumentId=current["sourceDocumentId"], target=target,
                        sourcePreservation={"receipt": pin, "identifierOccurrences": occurrences, "accounting": signature})
        return rendered
