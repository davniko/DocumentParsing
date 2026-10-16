"""Source-owned commercial dependencies for full-scenario synthesis.

Customs references in the pilot are fictional, generic trade references, not
claims to have generated legally valid national registration numbers. Audited
caption spans remove the original national-system claim when geography varies.
Public carrier boilerplate remains intact. Separately named private commercial
actors require explicit generated identity ownership; they are never inferred
to be the primary shipper merely because they appear in its continuation.
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
from document_ocr.synthesis.curated_dg_wording import proper_shipping_name
from document_ocr.synthesis.curated_hs import hs_digits, render_source_hs
from document_ocr.synthesis.curated_scenarios import ShipmentScenario
from document_ocr.synthesis.curated_templates import RenderRegion, SamplingBlueprint
from document_ocr.synthesis.generators import (
    DeterministicStream,
    generate_from_surface_pattern,
    surface_pattern,
)
from document_ocr.synthesis.phone_presentations import format_phone_presentation
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
    actions = {
        "text",
        "target",
        "identifier",
        "date",
        "phone",
        "transport_leg",
        "party_location",
        "dg_fact",
    } & recipe.keys()
    if len(actions) != 1:
        raise ValueError("auxiliary recipe must have exactly one rendering action")
    action = next(iter(actions))
    options = {
        "text": {"prefix", "suffix"},
        "target": {"prefix", "suffix"},
        "identifier": {"prefix", "canonical_identity", "format", "source_value"},
        "date": {"format", "publish_target"},
        "phone": {"channel", "canonical_identity", "prefix", "suffix", "country_code"},
        "transport_leg": {"source_voyage"},
        "party_location": {"field"},
        "dg_fact": {"prefix", "suffix"},
    }[action]
    if set(recipe) - {action} - options:
        raise ValueError("unsupported auxiliary recipe options")
    if action == "dg_fact" and recipe[action] != "properShippingName":
        raise ValueError("unsupported DG identity fact")
    if recipe.get("format") is not None and action == "identifier" and recipe["format"] != "source":
        raise ValueError("identifier formatting must preserve the source envelope")
    if "source_value" in recipe and (
        action != "identifier"
        or recipe.get("format") != "source"
        or not isinstance(recipe["source_value"], str)
        or not recipe["source_value"]
        or not recipe["source_value"].isascii()
        or not any(c.isalnum() for c in recipe["source_value"])
    ):
        raise ValueError("source_value requires a nonempty ASCII source-format identifier")
    if action == "phone" and recipe.get("channel", "phone") not in {"phone", "fax"}:
        raise ValueError("auxiliary contact channel must be phone or fax")
    if action == "phone" and ((recipe["phone"] == "source") != ("country_code" in recipe)):
        raise ValueError("source contact requires an explicit country code; sampled roles do not")
    if "country_code" in recipe and not re.fullmatch(r"[A-Z]{2}", recipe["country_code"]):
        raise ValueError("source contact country code requires ISO alpha-2")
    if "canonical_identity" in recipe and (
        not isinstance(recipe["canonical_identity"], str)
        or not re.fullmatch(r"[A-Z0-9]+", recipe["canonical_identity"])
    ):
        raise ValueError("canonical identifier identity requires uppercase ASCII letters/digits")
    if action == "party_location" and (
        not isinstance(recipe["party_location"], str)
        or not recipe["party_location"].startswith("documentPatch.parties.")
        or recipe.get("field") not in {"name", "country", "country_code"}
    ):
        raise ValueError("party location requires an explicit party role and location field")
    if action == "transport_leg" and (
        not isinstance(recipe["transport_leg"], str)
        or not recipe["transport_leg"].strip()
        or not isinstance(recipe.get("source_voyage"), str)
        or not recipe["source_voyage"].strip()
    ):
        raise ValueError("transport leg requires a role and exact source voyage")
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
        if (path := recipe.get("target")) and ".hsCodes[" in path:
            if path not in flat(target):
                raise ValueError(f"HS alias has no source target: {key}")
            regions = [
                replace(region, target_paths=tuple(sorted(set(region.target_paths) | {path})))
                if region.key == key
                else region
                for region in regions
            ]
        if path := recipe.get("publish_target"):
            if path in flat(target):
                raise ValueError(f"audited auxiliary publication target already exists: {path}")
            owners = [r for r in regions if r.key == key]
            if not owners or any(r.curated_key is not None for r in owners):
                raise ValueError("audited date publication needs an exact source-only owner")
            for owner in owners:
                parsed = datetime.strptime(owner.source, recipe.get("format", "%Y-%m-%d")).date()
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
    return replace(blueprint, target=target, regions=tuple(regions), historical_bindings=bindings)


def _render_recipe(
    recipe: dict,
    scenario: ShipmentScenario,
    target: dict,
    stream: DeterministicStream,
    original: str,
    shifts: set[int],
    vessels: tuple[str, ...] = (),
    *,
    source_country_code: str | None = None,
    literal_phone_prefix: str = "",
) -> str:
    if "party_location" in recipe:
        locality = scenario.party_localities[recipe["party_location"]]
        return getattr(locality, recipe["field"])
    if "transport_leg" in recipe:
        voyage = recipe["source_voyage"]
        if not original.endswith(" " + voyage):
            raise ValueError("transport leg source does not end with the declared voyage")
        original_vessel = original[: -len(voyage)].strip()
        main = target["documentPatch"].get("transport", {}).get("vesselName")
        choices = tuple(v for v in vessels if v not in {original_vessel, main})
        if not choices:
            raise ValueError("auxiliary transport leg has no distinct train-fit vessel support")
        leg_stream = stream.derive("transport-leg:" + recipe["transport_leg"])
        vessel = choices[leg_stream.derive("vessel").randbelow(len(choices))]
        new_voyage = generate_from_surface_pattern(
            pattern=surface_pattern(voyage),
            stream=leg_stream.derive("voyage"),
            additional_excluded=voyage,
        )
        return vessel + " " + new_voyage
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
        if via := scenario.route_locations.get("transshipmentPort"):
            substitutions.update(
                transshipment_port=via.name,
                transshipment_country=via.country,
                transshipment_code=via.country_code,
            )
        # A port identifier is not a country code or a GeoNames record ID.
        # Expose only actual registered UN/LOCODEs; unresolved requests fail below.
        for role, location in (
            ("origin", scenario.origin),
            ("destination", scenario.destination),
            ("transshipment", via),
        ):
            if location is not None and location.registry == "unlocode_wpi":
                if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{3}", location.registry_id):
                    raise ValueError("registered port has invalid UN/LOCODE")
                substitutions[role + "_unlocode"] = location.registry_id
        for key, replacement in substitutions.items():
            value = value.replace("{" + key + "}", replacement)
        if re.search(r"\{[^}]+\}", value):
            raise ValueError("unknown auxiliary interpolation key")
        return recipe.get("prefix", "") + value + recipe.get("suffix", "")
    if "dg_fact" in recipe:
        return recipe.get("prefix", "") + proper_shipping_name(scenario) + recipe.get("suffix", "")
    if "target" in recipe:
        value = flat(target).get(recipe["target"])
        if not isinstance(value, str) or not value:
            raise ValueError(f"auxiliary alias lacks sampled target: {recipe['target']}")
        if ".hsCodes[" in recipe["target"]:
            value = render_source_hs(original, hs_digits(value))
        return recipe.get("prefix", "") + value + recipe.get("suffix", "")
    if "identifier" in recipe:
        # Source punctuation variants share an identity by default. Only an
        # explicit reviewed alias may join occurrences whose actual digits differ.
        identity = recipe.get("canonical_identity")
        if identity is None:
            identity = re.sub(r"[^A-Z0-9]", "", original.upper())
        if recipe.get("format") == "source":
            return recipe.get("prefix", "") + _source_format_identifier(recipe, original, stream)
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
        scope = recipe["phone"]
        if scope == "source":
            country_code = recipe["country_code"]
        elif scope == "origin":
            locality = scenario.origin
        elif scope == "destination":
            locality = scenario.destination
        else:
            locality = scenario.party_localities[scope]
        if scope != "source":
            country_code = locality.country_code
        generator = contact_values.fax if recipe.get("channel") == "fax" else contact_values.phone
        value = generator(
            stream.derive(
                "aux-phone:" + recipe.get("canonical_identity", re.sub(r"\D", "", original))
            ),
            country_code=country_code,
        )
        prefix, suffix = recipe.get("prefix", ""), recipe.get("suffix", "")
        core = original
        if prefix and core.startswith(prefix):
            core = core[len(prefix) :]
        if suffix and core.endswith(suffix):
            core = core[: -len(suffix)]
        presentation = format_phone_presentation(
            source_value=literal_phone_prefix + core,
            source_surfaces=(core,),
            replacement_identity=value,
            target_country_code=country_code,
            source_country_code=(country_code if scope == "source" else source_country_code),
            literal_prefixes=(literal_phone_prefix,),
        )
        return prefix + presentation.surfaces[0] + suffix
    raise ValueError(f"unknown auxiliary recipe: {recipe}")


def _source_format_identifier(recipe: dict, original: str, stream: DeterministicStream) -> str:
    """Replace an explicit code, retaining its case classes and literal envelope.

    A reviewed source_value can isolate a code inside an old mixed owner, such
    as an invoice reference followed by a date. Punctuation variants of one
    code share their canonical replacement without copying national prefixes.
    """
    source = recipe.get("source_value", original)
    if not source.isascii() or not any(c.isalnum() for c in source):
        raise ValueError("source-format identifier requires ASCII letters or digits")
    if source not in original:
        raise ValueError("source-format identifier source_value is absent from its owner")
    canonical = re.sub(r"[^A-Z0-9]", "", source.upper())
    identity = recipe.get("canonical_identity", canonical)
    # Different presentation separators/case cannot allocate another identity.
    if len(identity) != len(canonical) or any(
        a.isdigit() != b.isdigit() for a, b in zip(identity, canonical, strict=True)
    ):
        raise ValueError("canonical source identifier has incompatible character classes")
    derived = stream.derive("aux-source-id:" + identity)
    if canonical.isdigit():
        floor = 10 ** (len(canonical) - 1)
        value = str(floor + derived.randbelow(9 * floor))
        if value == canonical:
            value = str(floor + (int(value) - floor + 1) % (9 * floor))
    elif len(identity) == 32 and re.fullmatch(r"[A-F0-9]{32}", identity):
        # UUID/128-bit hexadecimal references remain hexadecimal, including
        # punctuation-free aliases. Casing and separators are applied below.
        for attempt in range(10000):
            value = "".join(
                alphabet[
                    derived.derive(f"character-{index}").randbelow(len(alphabet), counter=attempt)
                ]
                for index, c in enumerate(identity)
                for alphabet in ("0123456789" if c.isdigit() else "ABCDEF",)
            )
            if value != canonical:
                break
        else:
            raise ValueError("hexadecimal source identifier space is exhausted")
    else:
        value = generate_from_surface_pattern(
            pattern=surface_pattern(identity), stream=derived, additional_excluded=canonical
        )
    characters = iter(value)
    rendered = "".join(
        next(characters).lower() if c.islower() else next(characters) if c.isalnum() else c
        for c in source
    )
    if rendered == source:
        raise ValueError("source-format identifier retained the original identity")
    return original.replace(source, rendered)


def auxiliary_surfaces(
    blueprint: SamplingBlueprint,
    scenario: ShipmentScenario,
    target: dict,
    stream: DeterministicStream,
    *,
    existing: dict[str, str | list[str]] | None = None,
    vessels: tuple[str, ...] = (),
    source_country_codes: dict[str, str] | None = None,
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
    legs: dict[str, str] = {}
    for key, binding in blueprint.historical_bindings.items():
        recipe = binding.get("auxiliary_recipe", {})
        if key not in present_keys or "transport_leg" not in recipe:
            continue
        role = recipe["transport_leg"]
        for occurrence in binding["occurrences"]:
            value = occurrence["source_text"]
            if role in legs and legs[role] != value:
                raise ValueError(
                    "one transport leg has conflicting source vessel/voyage declarations"
                )
            legs[role] = value
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
            if public_paths[key] and set(recipe) & {"identifier", "phone", "transport_leg"}:
                raise ValueError(
                    f"source-only auxiliary generator owns public target values: {key}"
                )
            rendered = []
            for index, original in enumerate(originals):
                literal_prefix = ""
                if "phone" in recipe:
                    owners = [
                        region for region in blueprint.regions
                        if region.key == key and region.occurrence == index
                    ]
                    if len(owners) != 1:
                        raise ValueError("source-only phone requires one exact occurrence owner")
                    raw, start = blueprint.source.encode(), owners[0].start
                    if start and raw[start - 1 : start] == b"+":
                        literal_prefix = "+"
                rendered.append(
                    _render_recipe(
                        recipe, scenario, target, stream, original, shifts, vessels,
                        source_country_code=(source_country_codes or {}).get(recipe.get("phone")),
                        literal_phone_prefix=literal_prefix,
                    )
                )
        else:
            supplied = output.get(key, originals)
            rendered = list(supplied) if isinstance(supplied, list) else [supplied] * len(originals)
        for transform in transformations:
            # Transform once per occurrence, even if compile saw repeats.
            replacement = _render_recipe(
                {"text": transform["text"]},
                scenario,
                target,
                stream,
                transform["source"],
                shifts,
                vessels,
            )
            rendered = [text.replace(transform["source"], replacement) for text in rendered]
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
            if all(".hsCodes[" in path for path in public_paths[key]):
                expected = {hs_digits(value) for value in expected}
                actual = {hs_digits(value) for value in actual}
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


def validate_identifier_privacy(blueprint: SamplingBlueprint) -> None:
    """Require an explicit private/public/dependent disposition before generation.

    A carrier SCAC, form code or registration constant is not a private invoice.
    Those distinctions belong to reviewed metadata, not runtime key heuristics.
    Target-bearing IDs retain the existing host/alias ownership contract.
    """
    policy = blueprint.ownership_data.get("identifier_privacy")
    if not isinstance(policy, dict):
        raise ValueError("source-only identifiers require an explicit privacy inventory")
    if any(
        not isinstance(k, str)
        or not isinstance(v, str)
        or v not in {"private", "public", "dependent"}
        for k, v in policy.items()
    ):
        raise ValueError("identifier privacy requires named private/public/dependent owners")
    bindings = blueprint.historical_bindings
    if unknown := set(policy) - set(bindings):
        raise ValueError(f"identifier privacy names an unknown source owner: {sorted(unknown)}")
    leaves = flat(blueprint.target)
    # Dropped reference fields retain their original binding-role provenance.
    # Require disposition even when an old compiler called that owner other_text.
    eligible = {
        region.key
        for region in blueprint.regions
        if region.curated_key is None
        and not any(path in leaves for path in region.target_paths)
        and (
            region.kind == "identifier"
            or any(
                path == "documentPatch.forwardingAndExportReferences"
                or path.startswith("documentPatch.forwardingAndExportReferences[")
                for path in (
                    *bindings[region.key].get("target_paths", []),
                    *(
                        value["target_path"]
                        for value in bindings[region.key]
                        .get("realization", {})
                        .get("target_values", [])
                    ),
                )
            )
        )
    }
    if missing := eligible - set(policy):
        raise ValueError(f"source-only identifier privacy is unresolved: {sorted(missing)}")
    for key in eligible:
        recipe = bindings[key].get("auxiliary_recipe", {})
        if policy[key] == "private" and (
            "identifier" not in recipe or recipe.get("format") != "source"
        ):
            raise ValueError(f"private identifier requires a source-format generator: {key}")
        if policy[key] == "dependent" and not set(recipe) & {
            "target",
            "date",
            "phone",
            "party_location",
            "transport_leg",
            "text",
        }:
            raise ValueError(f"dependent identifier lacks its explicit sampled owner: {key}")
