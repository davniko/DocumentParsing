"""Reviewed, occurrence-owned package presentation; independent of cargo sampling.

The registry supplies lexical alternatives. A source certificate supplies their
scope. Neither may change the package identity, quantity or nesting ledger.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest, flat
from document_ocr.synthesis.curated_scenarios import FilePin
from document_ocr.synthesis.curated_templates import wrap_owned_text

Style = Literal["full", "short", "optional"]
CATEGORY = re.compile(
    r"documentPatch\.goodsItemDetails\[\d+\]\.numberAndTypeOfPackages\[\d+\]\.typeCategory"
)
OPTIONAL = re.compile(r"\((?:S|ES)\)", re.I)


class PackagePresentationPolicy(BaseModel):
    """Style weights are conditional on reviewed slot and vocabulary eligibility."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    vocabulary: FilePin
    certificates: FilePin
    full_weight: float = Field(default=0.5, ge=0)
    short_weight: float = Field(default=0.3, ge=0)
    optional_weight: float = Field(default=0.2, ge=0)

    @model_validator(mode="after")
    def positive(self):
        if self.full_weight <= 0:
            raise ValueError("full wording must remain eligible for specificity witnesses")
        return self

    def weights(self):
        return {
            "full": self.full_weight,
            "short": self.short_weight,
            "optional": self.optional_weight,
        }


class PackageVocabulary:
    """Pinned category-to-presentation pools; no global text substitutions."""

    def __init__(self, payload: dict[str, Any]):
        if payload.get("version") != 1:
            raise ValueError("unsupported package presentation vocabulary")
        self.entries = payload["entries"]
        self.excluded = frozenset(payload["excludedUnscopedTokens"])
        owners: dict[str, set[str]] = defaultdict(set)
        for category, entry in self.entries.items():
            if set(entry) != {"full", "short", "optional", "family"} or not entry["full"]:
                raise ValueError("package vocabulary requires full/short/optional/family")
            if entry["family"] not in self.entries:
                raise ValueError("unknown package presentation family")
            for style in ("full", "short", "optional"):
                forms = entry[style]
                if not isinstance(forms, list) or len(forms) != len(set(forms)):
                    raise ValueError("package forms must be a distinct explicit list")
                for value in forms:
                    if not isinstance(value, str) or not value or value != value.strip().upper():
                        raise ValueError("package forms must be nonempty canonical uppercase")
                    if value in self.excluded:
                        raise ValueError("quarantined package alias")
                    if style == "optional" and OPTIONAL.search(value) is None:
                        raise ValueError("optional pool lacks explicit optional inflection")
                    owners[value].add(category)
        if any(len(categories) > 1 for categories in owners.values()):
            raise ValueError("unscoped alias has conflicting package identities")

    def forms(self, category: str, style: Style) -> tuple[str, ...]:
        if category not in self.entries:
            raise ValueError(f"package category lacks reviewed presentations: {category}")
        return tuple(self.entries[category][style])

    def supports(self, text: str, category: str) -> bool:
        text = " ".join(text.split()).upper()
        return any(
            re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text)
            for style in ("full", "short", "optional")
            for value in self.forms(category, style)
        )

    def choose(
        self,
        category: str,
        styles: tuple[Style, ...],
        weights: dict[str, float],
        seed: str,
        *,
        maximum_advance: float | None = None,
        metrics=None,
    ):
        pools = {}
        exclusions = {}
        for style in styles:
            forms = self.forms(category, style)
            eligible = tuple(
                v
                for v in forms
                if maximum_advance is None or metrics.length(v) <= maximum_advance + 1e-7
            )
            if eligible and weights[style] > 0:
                pools[style] = eligible
            else:
                exclusions[style] = "no_fitting_reviewed_form" if forms else "no_reviewed_form"
        if not pools:
            raise ValueError(f"no eligible package presentation for {category}")
        rng = random.Random(seed)
        style = rng.choices(tuple(pools), weights=[weights[s] for s in pools], k=1)[0]
        total = sum(weights[s] for s in pools)
        return rng.choice(pools[style]), {
            "style": style,
            "eligibleProbabilities": {s: weights[s] / total for s in pools},
            "excludedStyles": exclusions,
        }


class PackagePresentations:
    """Shared campaign instance; all per-document decisions use local RNGs."""

    def __init__(self, root: Path, policy: PackagePresentationPolicy, metrics):
        self.policy, self.metrics = policy, metrics
        for pin in (policy.vocabulary, policy.certificates):
            if sha256_file(root / pin.path) != pin.sha256:
                raise ValueError("package presentation authority hash mismatch")
        self.vocabulary = PackageVocabulary(json.loads((root / policy.vocabulary.path).read_text()))
        certificates = json.loads((root / policy.certificates.path).read_text())
        if certificates.get("version") != 1:
            raise ValueError("unsupported package occurrence certificates")
        self.certification_status = certificates.get("status")
        self.acceptance = certificates.get("acceptance", {})
        self.certificates = certificates["sources"]

    def require_certified(self, source_ids):
        """Inspection authorities can render probes, never publish training data."""
        if self.certification_status != "certified_native_synthesis_and_geometry":
            raise ValueError("package presentations are not production certified")
        if any(self.acceptance.get(sid, {}).get("status") != "PASS" for sid in source_ids):
            raise ValueError("selected source lacks package presentation acceptance")

    def certificate(self, blueprint):
        try:
            certificate = self.certificates[blueprint.document_id]
        except KeyError as error:
            raise ValueError("source has no package presentation certificate") from error
        if certificate["sourceSha256"] != digest(blueprint.source.encode()):
            raise ValueError("package presentation source certificate is stale")
        if certificate["sourceTargetSha256"] != digest(blueprint.target):
            raise ValueError("package presentation source target certificate is stale")
        actual = {
            (r.key, r.occurrence): r
            for r in blueprint.regions
            if any(CATEGORY.fullmatch(p) for p in r.target_paths)
        }
        expected = {(s["key"], s["occurrence"]): s for s in certificate["slots"]}
        if actual.keys() != expected.keys():
            raise ValueError("package presentation ownership coverage changed")
        for identity, region in actual.items():
            slot = expected[identity]
            if (slot["start"], slot["end"], slot["source"]) != (
                region.start,
                region.end,
                region.source,
            ):
                raise ValueError("package presentation source span changed")
            if slot["categoryPaths"] != [p for p in region.target_paths if CATEGORY.fullmatch(p)]:
                raise ValueError("package presentation target ownership changed")
        return certificate

    def noun_renderer(self, source_id: str, seed: int | str, variant: int | str):
        """Called by the expression constructor, never on a finished string.

        Compound levels always retain their complete category; contextual family
        shortening is reserved for explicitly associated scalar repetitions.
        """
        if source_id not in self.certificates:
            raise ValueError("compound source has no alias certificate")

        def noun(category: str, quantity: int) -> str:
            if type(quantity) is not int or quantity < 1:
                raise ValueError("compound noun needs a positive counted quantity")
            value, _ = self.vocabulary.choose(
                category,
                ("full", "short", "optional"),
                self.policy.weights(),
                f"package-compound:{seed}:{source_id}:{variant}:{category}:{quantity}",
            )
            return value

        return noun

    def locality_projection_view(self, original, projected):
        """Keep alias certificates strict across an authorized country omission."""
        from copy import copy

        from document_ocr.synthesis.independent_geography import validate_locality_target_projection

        certificate = self.certificate(original)
        omitted = validate_locality_target_projection(original, projected)
        view = copy(self)
        view.certificates = {
            **self.certificates,
            original.document_id: {
                **certificate,
                "sourceTargetSha256": digest(projected.target),
                "localityProjection": {
                    "originalCertificateSha256": digest(certificate), "omittedPaths": omitted,
                },
            },
        }
        # This rechecks every package owner, category path, source span and byte.
        view.certificate(projected)
        return view

    def apply(self, blueprint, target, surfaces, *, seed: int, variant: int):
        """Project reviewed scalar nouns after physical/complete-region ownership."""
        certificate = self.certificate(blueprint)
        leaves, original = flat(target), flat(blueprint.target)
        output = deepcopy(surfaces)
        slots = certificate["slots"]
        groups = defaultdict(list)
        for slot in slots:
            if slot["mode"] == "scalar":
                if len(slot["categoryPaths"]) != 1:
                    raise ValueError("scalar alias must own one package identity")
                groups[slot["categoryPaths"][0]].append(slot)
            elif slot["mode"] not in {"compound", "implicit"}:
                raise ValueError("unknown package presentation owner mode")
        evidence = []
        for path, owned in groups.items():
            category = leaves[path]
            family = self.vocabulary.entries[category]["family"]
            # Prefer a declaration, then the earliest associated noun. This is
            # an explicit complete witness; allocation repetitions may shorten.
            witness = min(owned, key=lambda s: (s["role"] != "declaration", s["start"]))
            for slot in owned:
                key, index = slot["key"], slot["occurrence"]
                binding = blueprint.historical_bindings[key]
                current = output.get(key, [o["source_text"] for o in binding["occurrences"]])
                values = (
                    list(current)
                    if isinstance(current, list)
                    else [current] * len(binding["occurrences"])
                )
                before = values[index]
                protected = slot.get("protectedSourceDetail", False) and category == original[path]
                if protected:
                    value = slot["source"]
                    choice = {
                        "style": "protected",
                        "eligibleProbabilities": {},
                        "excludedStyles": {},
                    }
                    presented = category
                else:
                    presented = (
                        family
                        if slot is not witness and slot["role"] in {"allocation", "repetition"}
                        else category
                    )
                    styles = ("full", "short", "optional")
                    if slot is witness and family != category:
                        styles = ("full", "optional")
                    # The shared page layout wraps the complete edited region
                    # within its source column. Do not narrow cargo categories
                    # merely because their full noun is longer than this source.
                    value, choice = self.vocabulary.choose(
                        presented,
                        styles,
                        self.policy.weights(),
                        f"package-slot:{seed}:{blueprint.document_id}:{variant}:{key}:{index}",
                    )
                    value = wrap_owned_text(slot["source"], value)
                values[index] = value
                output[key] = values
                evidence.append(
                    {
                        "key": key,
                        "occurrence": index,
                        "categoryPath": path,
                        "category": category,
                        "presentedCategory": presented,
                        "role": slot["role"],
                        "source": slot["source"],
                        "before": before,
                        "after": value,
                        "geometry": slot["geometry"],
                        **choice,
                    }
                )
            witnesses = [
                e
                for e in evidence
                if e["categoryPath"] == path
                and (self.vocabulary.supports(e["after"], category) or e["style"] == "protected")
            ]
            if family != category and not witnesses:
                raise ValueError("specific package identity lost its owned witness")
        return output, {
            "vocabularySha256": self.policy.vocabulary.sha256,
            "certificatesSha256": self.policy.certificates.sha256,
            "slots": evidence,
            "nonScalarOwners": [
                {"key": s["key"], "occurrence": s["occurrence"], "mode": s["mode"]}
                for s in slots
                if s["mode"] != "scalar"
            ],
        }
