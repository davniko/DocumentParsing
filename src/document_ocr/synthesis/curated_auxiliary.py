"""Source-owned commercial dependencies for full-scenario synthesis.

Customs references in the pilot are fictional, generic trade references, not
claims to have generated legally valid national registration numbers. Audited
caption spans remove the original national-system claim when geography varies.
Carrier boilerplate and unrelated third-party commercial actors remain intact.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from document_ocr.synthesis.curated import digest, flat
from document_ocr.synthesis.curated_scenarios import ShipmentScenario
from document_ocr.synthesis.curated_templates import RenderRegion, SamplingBlueprint
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.template_compiler import contact_values

# These are captions, not a whole-document country substitution. Exact matched
# byte spans are added to the same non-overlapping renderer as shipment facts.
_CAPTIONS = (
    (r"\bACID\s+COMPANY\s+VAT\s+NUMBER", "IMPORTER REGISTRATION NUMBER"),
    (r"\b(?:EG\s+)?ACID-Advance Cargo information declaration", "IMPORT REFERENCE"),
    (
        r"\b(?:HOUSE\s+|EG\s+)?ACID(?:"
        r"(?:\s+REGISTRATION\s+NUMBER|\s+NUMBER|\s+NO\.?)(?=\s*[:#.-]?\s*\d)"
        r"|(?=\s*[:#]\s*\d))",
        "IMPORT REFERENCE",
    ),
    (r"\bACI\s+NO\.", "IMPORT REFERENCE"),
    (
        r"\bEGYPTIAN\s+IMPORTER\s+(?:VAT\s+ID\s*/\s*TAX\s+ID|VAT\s+NUMBER|TAX\s+ID)",
        "IMPORTER REGISTRATION NUMBER",
    ),
    (r"\bCNPJ/CPF", "SHIPPER REGISTRATION"),
    (r"\bNIP(?=\s*:)", "SHIPPER REGISTRATION"),
    (r"\bCIF(?=\s*:)", "REGISTRATION"),
    (r"\bABN(?=\s+\d)", "EXPORTER REGISTRATION"),
    (r"\bRUC(?=\s*:)", "EXPORT REFERENCE"),
    (r"\bIBAN\(EUR\)", "BANK REFERENCE (EUR)"),
    (r"\bSEAL\s+SIF(?=\s*:)", "SEAL"),
    (r"\bITC\s+HS\s+CODE(?=\s*:)", "HS CODE"),
    (r"\bHTS(?=\s*:)", "HS CODE"),
)


def load_auxiliary_contract(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if value.get("version") != 1 or not isinstance(value.get("sources"), dict):
        raise ValueError("unsupported auxiliary dependency contract")
    for source in value["sources"].values():
        if set(source) - {"bindings", "spans"}:
            raise ValueError("unknown auxiliary source declaration")
        recipes = list(source.get("bindings", {}).values())
        for span in source.get("spans", []):
            if set(span) != {"quote", "count", "value"} or not span["quote"] or span["count"] < 1:
                raise ValueError("invalid auxiliary exact-span declaration")
            recipes.append(span["value"])
        for recipe in recipes:
            _validate_recipe(recipe)
    return value


def _validate_recipe(recipe: dict) -> None:
    # YAML single-quoted strings do not decode escapes. These values become
    # document text, so an escaped line break would silently corrupt layout.
    for key in ("text", "prefix", "suffix"):
        value = recipe.get(key, "")
        if not isinstance(value, str) or re.search(r"\\[nrt]|\r", value):
            raise ValueError(f"auxiliary {key} requires plain text with real LF line breaks")
    actions = {"text", "target", "identifier", "date", "phone"} & recipe.keys()
    if len(actions) != 1:
        raise ValueError("auxiliary recipe must have exactly one rendering action")
    action = next(iter(actions))
    options = {
        "text": {"prefix", "suffix"},
        "target": {"prefix", "suffix"},
        "identifier": {"prefix"},
        "date": {"format", "publish_target"},
        "phone": set(),
    }[action]
    if set(recipe) - {action} - options:
        raise ValueError("unsupported auxiliary recipe options")
    if action == "date":
        date.fromisoformat(recipe["date"])
        if recipe.get("publish_target") not in (
            None,
            "documentPatch.issueDate",
            "documentPatch.shippedOnBoardDate",
        ):
            raise ValueError("unsupported audited date publication target")


def _source_spec(blueprint: SamplingBlueprint, contract: dict) -> dict:
    matches = [
        v
        for k, v in contract["sources"].items()
        if blueprint.document_id == k or blueprint.document_id.removeprefix("doc_").startswith(k)
    ]
    if len(matches) != 1:
        raise ValueError(f"auxiliary source coverage must be unique: {blueprint.document_id}")
    return matches[0]


def augment_auxiliary_blueprint(blueprint: SamplingBlueprint, contract: dict) -> SamplingBlueprint:
    """Add certified caption/local-context spans; never expand into another owner."""
    spec = _source_spec(blueprint, contract)
    raw = blueprint.source.encode()
    regions = list(blueprint.regions)
    target = deepcopy(blueprint.target)
    bindings = deepcopy(dict(blueprint.historical_bindings))
    requests: list[tuple[str, int, int, str, dict]] = []
    selected: list[tuple[int, int]] = []
    for pattern, replacement in _CAPTIONS:
        for match in re.finditer(pattern, blueprint.source, re.I):
            begin = len(blueprint.source[: match.start()].encode())
            end = begin + len(match[0].encode())
            if any(begin < b and a < end for a, b in selected):
                continue  # Longest caption rule owns this already-selected text.
            selected.append((begin, end))
            requests.append((f"aux:caption:{begin}", begin, end, match[0], {"text": replacement}))
    for index, declaration in enumerate(spec.get("spans", [])):
        if "publish_target" in declaration["value"]:
            raise ValueError("audited date publication requires a historical date binding")
        quote = declaration["quote"]
        matches = list(re.finditer(re.escape(quote), blueprint.source))
        if len(matches) != declaration["count"]:
            raise ValueError(
                f"auxiliary exact quote occurrence mismatch: {blueprint.document_id}/{quote!r}"
            )
        for occurrence, match in enumerate(matches):
            begin = len(blueprint.source[: match.start()].encode())
            end = begin + len(quote.encode())
            requests.append(
                (f"aux:declared:{index}:{occurrence}", begin, end, quote, declaration["value"])
            )
    for key, begin, end, source, recipe in requests:
        overlaps = [r for r in regions if begin < r.end and r.start < end]
        if overlaps:
            # An entire old auxiliary binding may contain its caption (e.g. a
            # ventilation footer ending with ACID). Record a scoped transform
            # for that owner instead of introducing an overlapping edit.
            if (
                len(overlaps) == 1
                and overlaps[0].start <= begin
                and end <= overlaps[0].end
                and "text" in recipe
            ):
                owner = overlaps[0]
                if owner.curated_key is not None:
                    raise ValueError(
                        f"auxiliary caption intersects generated lexical owner: {owner.key}"
                    )
                binding = bindings[owner.key]
                binding.setdefault("auxiliary_caption_transforms", []).append(
                    {"source": source, "text": recipe["text"]}
                )
                continue
            raise ValueError(
                f"auxiliary span intersects another owner: {key}/{[r.key for r in overlaps]}"
            )
        if raw[begin:end].decode() != source:
            raise ValueError("auxiliary byte identity mismatch")
        regions.append(RenderRegion(key, 0, begin, end, source, (), "auxiliary"))
        bindings[key] = {
            "logical_key": key,
            "target_paths": [],
            "value_kind": "other_text",
            "occurrences": [{"byte_start": begin, "byte_end": end, "source_text": source}],
            "auxiliary_recipe": recipe,
        }
    regions.sort(key=lambda r: (r.start, r.end))
    if any(a.end > b.start for a, b in pairwise(regions)):
        raise ValueError("auxiliary augmentation introduced overlapping regions")
    for key in spec.get("bindings", {}):
        if key not in bindings:
            raise ValueError(f"unknown audited auxiliary binding: {key}")
        recipe = spec["bindings"][key]
        _validate_recipe(recipe)
        bindings[key]["auxiliary_recipe"] = recipe
        if path := recipe.get("publish_target"):
            if path in flat(target):
                raise ValueError(f"audited auxiliary publication target already exists: {path}")
            owners = [r for r in regions if r.key == key]
            if not owners or any(r.curated_key is not None for r in owners):
                raise ValueError("audited date publication needs an exact source-only owner")
            for owner in owners:
                parsed = datetime.strptime(
                    owner.source, recipe.get("format", "%Y-%m-%d")
                ).date()
                if (
                    parsed.isoformat() != recipe["date"]
                    or parsed.strftime(recipe.get("format", "%Y-%m-%d")) != owner.source
                ):
                    raise ValueError(f"audited date publication differs from source: {key}")
            # This is the narrowly authorized, source-proven addition above;
            # ordinary sampled target assignment remains topology-preserving.
            target["documentPatch"][path.rsplit(".", 1)[1]] = recipe["date"]
            regions = [
                replace(r, target_paths=tuple(sorted(set(r.target_paths) | {path})))
                if r.key == key
                else r
                for r in regions
            ]
    return replace(
        blueprint, target=target, regions=tuple(regions), historical_bindings=bindings
    )


def _render_recipe(
    recipe: dict,
    scenario: ShipmentScenario,
    target: dict,
    stream: DeterministicStream,
    original: str,
    shifts: set[int],
) -> str:
    if "text" in recipe:
        value = recipe["text"]
        substitutions = {
            "origin_country": scenario.origin.country,
            "origin_code": scenario.origin.country_code,
            "origin_port": scenario.origin.name,
            "destination_country": scenario.destination.country,
            "destination_code": scenario.destination.country_code,
            "destination_port": scenario.destination.name,
        }
        for key, replacement in substitutions.items():
            value = value.replace("{" + key + "}", replacement)
        if re.search(r"\{[^}]+\}", value):
            raise ValueError("unknown auxiliary interpolation key")
        return recipe.get("prefix", "") + value + recipe.get("suffix", "")
    if "target" in recipe:
        value = flat(target).get(recipe["target"])
        if not isinstance(value, str) or not value:
            raise ValueError(f"auxiliary alias lacks sampled target: {recipe['target']}")
        return recipe.get("prefix", "") + value + recipe.get("suffix", "")
    if "identifier" in recipe:
        # Same source identifier, even with hyphens/spaces, shares one new value.
        identity = re.sub(r"[^A-Z0-9]", "", original.upper())
        length = max(6, len(re.sub(r"\D", "", identity)))
        digits = str(
            10 ** (length - 1)
            + stream.derive("aux-id:" + identity).randbelow(9 * 10 ** (length - 1))
        )
        prefix = recipe.get("prefix", "")
        return prefix + digits
    if "date" in recipe:
        if len(shifts) != 1:
            raise ValueError("auxiliary date needs one coherent shipment date shift")
        value = date.fromisoformat(recipe["date"]) + timedelta(days=next(iter(shifts)))
        return value.strftime(recipe.get("format", "%Y-%m-%d"))
    if "phone" in recipe:
        locality = scenario.party_localities[recipe["phone"]]
        return contact_values.phone(
            stream.derive("aux-phone:" + re.sub(r"\D", "", original)),
            country_code=locality.country_code,
        )
    raise ValueError(f"unknown auxiliary recipe: {recipe}")


def auxiliary_surfaces(
    blueprint: SamplingBlueprint,
    scenario: ShipmentScenario,
    target: dict,
    stream: DeterministicStream,
    *,
    existing: dict[str, str | list[str]] | None = None,
) -> tuple[dict[str, str | list[str]], list[dict]]:
    """Produce all declared dependent surfaces and auditable semantic receipts.

    Call after physical surfaces so nested captions are transformed on their
    final value. Undeclared carrier/legal bindings are intentionally unchanged.
    """
    output = dict(existing or {})
    before, after = flat(blueprint.target), flat(target)
    shifts = {
        (date.fromisoformat(after[p]) - date.fromisoformat(v)).days
        for p, v in before.items()
        if p.endswith((".issueDate", ".shippedOnBoardDate")) and p in after
    }
    if not shifts:
        # Some OCR dates are source-only, excluded by the current task policy.
        # They still belong to the same explicitly sampled chronology stream.
        shifts = {30 + stream.derive("date-shift").randbelow(730)}
    receipts = []
    present_keys = {r.key for r in blueprint.regions}
    public_paths: dict[str, set[str]] = {}
    for region in blueprint.regions:
        public_paths.setdefault(region.key, set()).update(
            p for p in region.target_paths if p in after
        )
    for key, binding in blueprint.historical_bindings.items():
        if key not in present_keys:
            continue  # A complete lexical owner supersedes this narrower span.
        recipe = binding.get("auxiliary_recipe")
        transformations = binding.get("auxiliary_caption_transforms", [])
        if not recipe and not transformations:
            continue
        originals = [o["source_text"] for o in binding["occurrences"]]
        if recipe:
            if key in output:
                raise ValueError(f"two semantic stages supplied auxiliary binding: {key}")
            if public_paths[key] and set(recipe) & {"identifier", "phone"}:
                raise ValueError(
                    f"source-only auxiliary generator owns public target values: {key}"
                )
            rendered = [
                _render_recipe(recipe, scenario, target, stream, original, shifts)
                for original in originals
            ]
        else:
            supplied = output.get(key, originals)
            rendered = list(supplied) if isinstance(supplied, list) else [supplied] * len(originals)
        for transform in transformations:
            # Transform once per occurrence, even if compile saw repeats.
            rendered = [text.replace(transform["source"], transform["text"]) for text in rendered]
        if any(not v.strip() for v in rendered):
            raise ValueError("auxiliary renderer produced an empty owned region")
        if recipe and public_paths[key]:
            # Explicit surfaces bypass the default scalar renderer. A public
            # leaf must therefore agree with its actual generated text, not
            # merely have an old binding claiming that it is covered.
            expected = {" ".join(str(after[p]).upper().split()) for p in public_paths[key]}
            actual = (
                {
                    datetime.strptime(v, recipe.get("format", "%Y-%m-%d")).date().isoformat()
                    for v in rendered
                }
                if "date" in recipe
                else {" ".join(v.upper().split()) for v in rendered}
            )
            if expected != actual:
                raise ValueError(
                    f"auxiliary surface disagrees with current public target: {key}; "
                    f"paths={sorted(public_paths[key])}"
                )
        output[key] = rendered if len(set(rendered)) > 1 else rendered[0]
        receipts.append(
            {
                "key": key,
                "recipe": recipe,
                "captionTransforms": transformations,
                "before": originals,
                "after": rendered,
            }
        )
    return output, receipts


def auxiliary_contract_receipt(contract: dict) -> dict:
    return {
        "contractSha256": digest(contract),
        "policy": "generic_fictional_trade_references_not_national_registration_validation",
    }
