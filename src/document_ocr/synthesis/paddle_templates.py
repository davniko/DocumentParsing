"""Pinned native-template presentation over newly sampled or retained shipments.

This layer does not choose cargo or infer new bindings. Existing shipment sampling
provides facts and lexical owners; the native catalog provides their reviewed
Paddle occurrences. Unknown owners fail instead of falling back to GLM text.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.spatial_inputs.paddle import fingerprint, pages_from_record
from document_ocr.synthesis.curated_equipment_aliases import EquipmentAliasRegistry
from document_ocr.synthesis.curated_reflow_policy import ReflowPolicy
from document_ocr.synthesis.package_presentations import PackageVocabulary
from document_ocr.synthesis.paddle_package_modes import validate_package_mode_owners
from document_ocr.synthesis.paddle_reflow import PaddleReflow
from document_ocr.synthesis.paddle_render import render_one, requires_wording
from document_ocr.synthesis.paddle_source_casing import POLICY as PRINTED_CASING_POLICY
from document_ocr.synthesis.paddle_source_casing import project_source_casing
from document_ocr.synthesis.paddle_validation import (
    check_conservation,
    check_new_private_ownership,
    check_private_lexemes,
    check_private_residue,
    check_synthetic_relations,
    check_views,
)


def verified_path(root: Path, pin: dict) -> Path:
    relative = Path(pin["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("catalog evidence must be project-relative")
    path = root / relative
    if sha256_file(path) != pin["sha256"]:
        raise ValueError(f"native template evidence changed: {relative}")
    return path


def read_pin(root: Path, pin: dict):
    return json.loads(verified_path(root, pin).read_text())


def validate_override_selectors(overlay: dict) -> None:
    """Reject ambiguous selectors, including ones unused by a replay shipment."""
    selectors_by_key: dict[str, list[tuple[int | None, int | None]]] = {}
    for override in overlay.get("overrides", []):
        if "softLineBreaks" in override and override["softLineBreaks"] is not True:
            raise ValueError("soft line breaks require explicit reviewed enablement")
        key, start = override["key"], override.get("byteStart")
        end = override.get("byteEnd")
        if end is not None and (
            type(start) is not int or type(end) is not int or not 0 <= start < end
        ):
            raise ValueError("bounded native selector requires a valid byte interval")
        selectors = selectors_by_key.setdefault(key, [])
        if any(
            (start is None or old_start is None or start == old_start)
            and (end is None or old_end is None or end == old_end)
            for old_start, old_end in selectors
        ):
            raise ValueError(f"ambiguous native override selector: {key} at {start}")
        selectors.append((start, end))


class PaddleTemplateCatalog:
    """Load one explicitly pinned catalog; cache only its finite admitted members."""

    def __init__(self, root: Path, manifest: dict):
        self.root = root.resolve()
        self.manifest = read_pin(self.root, manifest)
        self.target_casing = self.manifest.get("targetCasing")
        if self.target_casing not in (None, PRINTED_CASING_POLICY):
            raise ValueError("unsupported native target casing policy")
        if self.manifest["format"] != "native-paddle-templates-v1":
            raise ValueError("unsupported native template format")
        self.entries = self.manifest["templates"]
        if not self.entries or len(self.entries) != self.manifest["templateCount"]:
            raise ValueError("native template membership differs")
        self.validator = Draft202012Validator(read_pin(self.root, self.manifest["schema"]))
        self.equipment = EquipmentAliasRegistry.from_data(
            read_pin(self.root, self.manifest["equipment"])
        )
        self.packages = PackageVocabulary(read_pin(self.root, self.manifest["packages"]))
        self.engine = PaddleReflow(self.root, ReflowPolicy.model_validate(self.manifest["reflow"]))
        self._loaded = {}
        self.package_modes = self.manifest.get("declaredPackages")
        if self.package_modes is not None:
            self.package_contracts = read_pin(self.root, self.package_modes["contracts"])["sources"]
            if set(self.entries) - self.package_contracts.keys():
                raise ValueError("native package modes lack source contracts")
            if set(self.package_modes["singleRowOnly"]) - self.entries.keys():
                raise ValueError("single-row restriction names a nonmember template")

    def _template(self, source_id: str):
        if source_id not in self.entries:
            raise ValueError("shipment source is not in the native template catalog")
        if source_id not in self._loaded:
            entry = self.entries[source_id]
            template = read_pin(self.root, entry["template"])
            validate_override_selectors(template["overlay"])
            source = read_pin(self.root, template["source"])
            for positions in template["overlay"].get("sourceCaseRegions", {}).values():
                if not positions or len({tuple(r) for r in positions}) != len(positions):
                    raise ValueError("casing occurrence regions must be nonempty and unique")
                for page, region in positions:
                    if not (
                        0 <= page < len(source["pages"])
                        and 0 <= region < len(source["pages"][page]["regions"])
                    ):
                        raise ValueError("casing occurrence lies outside native source")
            transfer = read_pin(self.root, template["transfer"])
            if (
                self.package_modes is not None
                and source_id not in self.package_modes["singleRowOnly"]
            ):
                validate_package_mode_owners(
                    source_id, self.package_contracts[source_id], template, transfer
                )
            if template["sourceDocumentId"] != source_id or source["documentId"] != source_id:
                raise ValueError("native template/source identities differ")
            native = fingerprint(pages_from_record(source))
            if native != source["nativeFingerprint"] or transfer["source"] != source_id:
                raise ValueError("native transfer belongs to different Paddle input")
            # Source and transfer are jointly pinned by the reviewed template.
            # A redundant fingerprint, when recorded, must also agree.
            if "sourceFingerprint" in transfer and native != transfer["sourceFingerprint"]:
                raise ValueError("native transfer fingerprint differs")
            sampling_path = verified_path(self.root, entry["samplingSource"])
            sampling_source = sampling_path.read_bytes()
            self._loaded[source_id] = (template, source, transfer, sampling_source)
        return self._loaded[source_id]

    def shipment(self, candidate_path: Path):
        """Validate supplied sample identity and every source-text edit before rendering."""
        path = candidate_path.resolve()
        relative = path.relative_to(self.root)
        candidate = json.loads(path.read_text())
        source_id, identity = candidate["sourceDocumentId"], candidate["documentId"]
        if self.package_modes is not None and source_id in self.package_modes["singleRowOnly"]:
            goods = candidate["target"]["documentPatch"].get("goodsItemDetails", [])
            if any(len(g.get("numberAndTypeOfPackages", [])) > 1 for g in goods):
                raise ValueError(
                    "native source is explicitly certified for single-row packages only"
                )
        template, source, transfer, sampling_source = self._template(source_id)
        if (
            candidate["proof"]["sourceSha256"]
            != self.entries[source_id]["samplingSource"]["sha256"]
        ):
            raise ValueError("shipment uses a different sampling source")
        for edit in candidate["proof"]["edits"]:
            lo, hi = edit["byteStart"], edit["byteEnd"]
            if not 0 <= lo < hi <= len(sampling_source):
                raise ValueError("shipment edit lies outside its sampling source")
            if sampling_source[lo:hi].decode() != edit["before"]:
                raise ValueError("shipment edit does not match its sampling source")
        selection = {
            "id": identity,
            "path": str(relative),
            "receiptSha256": sha256_file(path),
        }
        # A new selection is local to this call; never modify certified controls.
        selected_transfer = {**transfer, "descendants": [selection]}
        return template, source, selected_transfer, candidate

    def wording_specification(self, candidate_path: Path):
        template, source, _, candidate = self.shipment(candidate_path)
        overlay = template["overlay"]
        if not requires_wording(overlay):
            return None
        return {
            "documentId": candidate["documentId"],
            "sourceFingerprint": source["nativeFingerprint"],
            "candidateSha256": sha256_file(candidate_path),
            "requested": overlay["newWording"],
            "patterns": overlay.get("newWordingPatterns", {}),
            "context": {
                "origin": candidate["scenario"]["origin"],
                "destination": candidate["scenario"]["destination"],
                "goods": candidate["scenario"]["goods_identities"],
                "printedGoods": candidate["target"]["documentPatch"].get("goodsItemDetails", []),
                "equipment": candidate["target"]["documentPatch"].get("containerInformation", []),
                "parties": candidate["target"]["documentPatch"].get("parties", {}),
                "partyLocalities": candidate["scenario"]["party_localities"],
            },
        }

    def render(self, candidate_path: Path, *, current=None, wording_receipt=None):
        """Return an audited candidate, never an automatically accepted training row."""
        template, source, transfer, candidate = self.shipment(candidate_path)
        identity = candidate["documentId"]
        if current is None:
            target = BillOfLadingExtractionV7DirectTarget.model_validate_json(
                json.dumps(candidate["target"]["documentPatch"])
            ).canonical_target()
            current = {"joinedRawText": candidate["joinedRawText"], "target": target}
        else:
            current = copy.deepcopy(current)
            if current["documentId"] != identity:
                # A reviewed replacement can retain the dataset identity while
                # using a fresh shipment receipt. Only the pinned transfer may
                # establish that relationship; callers cannot relabel receipts.
                accepted = self._template(candidate["sourceDocumentId"])[2]["descendants"]
                matches = [
                    s
                    for s in accepted
                    if s["id"] == current["documentId"]
                    and (self.root / s["path"]).resolve() == candidate_path.resolve()
                ]
                if len(matches) != 1:
                    raise ValueError("retained target belongs to a different shipment")
                identity = current["documentId"]
                transfer["descendants"][0]["id"] = identity
        rendered = render_one(
            source,
            transfer,
            candidate,
            current,
            template["overlay"],
            self.engine,
            identity,
            self.equipment,
            self.packages,
            wording_receipt=wording_receipt,
            project_root=self.root,
        )
        rendered = json.loads(json.dumps(rendered, ensure_ascii=False, allow_nan=False))
        if self.target_casing == PRINTED_CASING_POLICY:
            rendered = project_source_casing(rendered, candidate, template["overlay"], source)
        self.validator.validate(rendered["target"])
        check_views(rendered, pages_from_record(rendered))
        check_conservation(source, rendered, template["overlay"])
        check_synthetic_relations(rendered["target"])
        privacy = self.entries[source["documentId"]]["privacy"]
        if privacy["kind"] == "reviewed_native_owners":
            decision = privacy["decision"]
            check_private_lexemes(source, rendered, template["overlay"].get("retainedLexemes", []))
            check_new_private_ownership(source, rendered, decision)
        elif privacy["kind"] == "reviewed_replacement_spans":
            check_private_residue(read_pin(self.root, privacy["reference"]), rendered)
        else:
            raise ValueError("unknown native privacy authority")
        return rendered
