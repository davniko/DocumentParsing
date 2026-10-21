"""Country presentations over exact, already-rendered source ownership.

Canonical lexical generation and its validators run first. This layer changes
only proven country subspans, keeps normalized country identities, and projects
the corresponding printed address wording. It never searches a whole document.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from document_ocr.synthesis.country_presentations import CountryPresentationConfig, Registry
from document_ocr.synthesis.curated import digest, flat, layout_surface
from document_ocr.synthesis.curated_casing import RenderCasing, TargetCasing
from document_ocr.synthesis.generators import DeterministicStream

if TYPE_CHECKING:
    from document_ocr.synthesis.curated_templates import RenderRegion, SamplingBlueprint

_NAME_FAMILIES = ("name", "abbreviation", "localized_name")
_CODE_FAMILIES = ("iso2", "iso3", "dotted_code")
_CODE = re.compile(r"[A-Za-z](?:\.?[A-Za-z]){1,2}\.?")
_CONTACT_TOKEN = re.compile(r"\S+@\S+|(?:https?://|www\.)\S+", re.I)


def _fold(value: str) -> str:
    return " ".join(value.split())


def _phrase(value: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, value.split())) + r"(?!\w)", re.I)


def _fold_with_offsets(text: str) -> tuple[str, list[tuple[int, int]]]:
    parts, positions = [], []
    for match in re.finditer(r"\s+|\S+", text):
        if match[0].isspace():
            parts.append(" ")
            positions.append(match.span())
        else:
            parts.append(match[0])
            positions.extend((index, index + 1) for index in range(*match.span()))
    return "".join(parts), positions


def country_code_caption(text: str, start: int) -> bool:
    """An immutable, directly preceding code caption constrains presentation."""
    line = text[text.rfind("\n", 0, start) + 1 : start]
    return bool(re.search(r"\bCOUNTRY\s+CODE\s*[:=]?\s*$", line, re.I))


def country_frame(text: str, code: str, registry: Registry) -> dict[str, Any] | None:
    """Recognize a complete NAME(CODE) frame, not an atomic parenthesized name."""
    if registry.is_atomic_name(text, code):
        return None
    pattern = re.compile(
        r"(?P<name>.+?)(?P<between>\s*(?:-\s*)?\(\s*)"
        r"(?P<code>[A-Za-z](?:\.?[A-Za-z]){1,2}\.?)(?P<ending>\s*\))",
        re.DOTALL,
    )
    match = pattern.fullmatch(text)
    if match is None or _CODE.fullmatch(_fold(match["name"])):
        return None
    if registry.require(match["name"]) != code or registry.require(match["code"]) != code:
        raise ValueError("owned country frame joins different country identities")
    return {
        "name": match["name"],
        "between": match["between"],
        "code": match["code"],
        "ending": match["ending"],
        "codeWidth": len(match["code"].replace(".", "")),
    }


def owned_country_spans(
    text: str,
    code: str,
    registry: Registry,
    *,
    postal: bool = False,
    protected_names: Sequence[str] = (),
) -> tuple[tuple[int, int, str], ...]:
    """Find identity-certified country mentions inside one existing owner.

    Physical word wraps are matched through an offset-preserving whitespace
    view. Exact sampled locality names are excluded before country selection,
    so CUBA inside SANTIAGO DE CUBA does not become a country occurrence.
    Named countries take precedence over incidental short codes in postal text.
    """
    folded, positions = _fold_with_offsets(text)
    protected = [match.span() for name in protected_names for match in _phrase(name).finditer(text)]
    spans = []
    for start, end, _ in registry.find_spans(folded, code, postal=postal):
        lo, hi = positions[start][0], positions[end - 1][1]
        parenthetical = re.match(r"\s*\([^()]+\)", text[hi:])
        if parenthetical and registry.is_atomic_name(text[lo : hi + parenthetical.end()], code):
            hi += parenthetical.end()
        if any(
            left <= lo and hi <= right and (left, right) != (lo, hi) for left, right in protected
        ):
            continue
        if spans and lo < spans[-1][1]:
            continue
        spans.append((lo, hi, text[lo:hi]))
    merged = []
    for span in spans:
        if merged:
            left = merged[-1]
            gap = text[left[1] : span[0]]
            close = re.match(r"\s*\)", text[span[1] :])
            if (
                not _CODE.fullmatch(_fold(left[2]))
                and _CODE.fullmatch(_fold(span[2]))
                and close
                and re.fullmatch(r"\s*(?:-\s*)?\(\s*", gap)
            ):
                end = span[1] + close.end()
                combined = text[left[0] : end]
                country_frame(combined, code, registry)
                merged[-1] = left[0], end, combined
                continue
        merged.append(span)
    independent = [
        span
        for span in merged
        if not any(_fold(span[2]).casefold() == _fold(name).casefold() for name in protected_names)
    ]
    if independent:
        merged = independent
    if not postal:
        if len(merged) > 1 and independent:
            raise ValueError("direct country owner has multiple independent country mentions")
        return tuple(merged)
    named = [span for span in merged if not _CODE.fullmatch(_fold(span[2]))]
    if named:
        # Prefer a complete country component over an alias embedded in a
        # different address component: SAN SALVADOR DEPT is not the separately
        # printed EL SALVADOR country. Delimiters inside an atomic country
        # presentation remain inside the match and do not split its identity.
        complete = []
        postal_caption_witnesses = []
        for span in named:
            lo, hi, _ = span
            prefix = re.split(r"[,;\n]", text[:lo])[-1]
            suffix = re.split(r"[,;\n]", text[hi:])[0]
            postal_caption = r"(?:ZIP(?:\s+CODE)?|CEP|POSTAL\s+CODE|POSTCODE)"
            postal_suffix = bool(
                re.fullmatch(r"\s*(?:" + postal_caption + r"\s*[:#]?\s*)?\d[\w -]*", suffix, re.I)
            )
            if not prefix.strip(" \t.()") and (not suffix.strip(" \t.()") or postal_suffix):
                complete.append(span)
            if re.fullmatch(r"\s*" + postal_caption + r"\s*[:#]?\s*\d[\w -]*", suffix, re.I):
                postal_caption_witnesses.append(span)
        if complete:
            return tuple(
                span for span in named if span in complete or span in postal_caption_witnesses
            )
        return tuple(named)
    qualified = []
    for span in merged:
        lo, hi, _ = span
        line_prefix = text[text.rfind("\n", 0, lo) + 1 : lo]
        line_tail = text[hi:].split("\n", 1)[0]
        # An immediately attached identifier suffix is not a postal-country
        # witness. In particular, BR-116 is a highway, not a country prefix.
        if re.match(r"[-/]\d", line_tail):
            continue
        remainder = line_tail.strip(" \t.,;)]")
        component_end = not remainder
        postcode = bool(re.fullmatch(r"(?:ZIP\s*:?\s*)?\d[\w -]*", remainder, re.I))
        isolated = not line_prefix.strip(" \t,;:(-")
        if component_end or postcode or isolated:
            qualified.append(span)
    if len(qualified) > 1:
        raise ValueError("postal owner has multiple possible country-code occurrences")
    return tuple(qualified)


def bind_postal_country_alias_owners(
    blueprint: SamplingBlueprint, registry: Registry
) -> SamplingBlueprint:
    """Ground existing normalized countries in their complete postal alias owners.

    A normalized target need not literally occur in its source address. Keep
    already reviewed country owners intact; add only a missing same-party link
    witnessed by the pinned country registry inside a complete postal region.
    No target, printed text, source span or geographic interpretation changes.
    """
    before = flat(blueprint.target)
    owned = {path for region in blueprint.regions for path in region.target_paths}
    missing = {
        path
        for path in before
        if path.startswith("documentPatch.parties.") and path.endswith(".country")
    } - owned
    if not missing:
        return blueprint
    postal_variables = {
        variable.key for variable in blueprint.contract.variables if variable.kind == "postal"
    }
    regions = []
    changed = False
    for region in blueprint.regions:
        paths = set(region.target_paths)
        for address_path in region.target_paths:
            if not address_path.endswith(".addressLine"):
                continue
            country_path = address_path.removesuffix(".addressLine") + ".country"
            if country_path not in missing:
                continue
            if region.curated_key not in postal_variables and _fold(region.source).casefold() != (
                _fold(before[address_path]).casefold()
            ):
                continue
            code = registry.require(before[country_path], field=country_path)
            contacts = [match.span() for match in _CONTACT_TOKEN.finditer(region.source)]
            witnesses = owned_country_spans(region.source, code, registry, postal=True)
            if any(
                not any(start < hi and end > lo for lo, hi in contacts)
                for start, end, _ in witnesses
            ):
                paths.add(country_path)
        if paths != set(region.target_paths):
            region = replace(region, target_paths=tuple(sorted(paths)))
            changed = True
        regions.append(region)
    return replace(blueprint, regions=tuple(regions)) if changed else blueprint


def _case_surface(surface: str, original: str, family: str, style: RenderCasing) -> str:
    if family in (*_CODE_FAMILIES, "abbreviation"):
        return surface.upper()
    if style == "uppercase" or (style == "preserve" and original.isupper()):
        return surface.upper()
    if style == "title" or (style == "preserve" and original.istitle()):
        return surface.title()
    if style != "preserve":
        raise ValueError(f"unknown country render casing: {style}")
    return surface


def terminal_country_conflict(
    text: str,
    code: str,
    registry: Registry,
    *,
    protected_names: Sequence[str] = (),
) -> dict[str, Any] | None:
    """Detect a foreign suffix supported only by a city-as-country match.

    An independent own-country name is valid evidence even when a territorial
    affiliation follows it (RÉUNION, FRANCE). A protected city word alone is
    not: DJIBOUTI city cannot validate a postal ending in FALKLAND ISLANDS.
    """
    folded, positions = _fold_with_offsets(text)
    terminal = registry.terminal_named_country(folded)
    if terminal is None or terminal[3] == code:
        return None
    start, end, _, actual = terminal
    lo, hi = positions[start][0], positions[end - 1][1]
    if any(
        match.start() <= lo and hi <= match.end()
        for name in protected_names
        for match in _phrase(name).finditer(text)
    ):
        return None
    witnesses = owned_country_spans(
        text, code, registry, postal=True, protected_names=protected_names
    )
    if any(
        not _CODE.fullmatch(_fold(surface))
        and not any(_fold(surface).casefold() == _fold(name).casefold() for name in protected_names)
        for _, _, surface in witnesses
    ):
        return None
    return {"start": lo, "end": hi, "surface": text[lo:hi], "code": actual}


def present_owned_country(
    text: str,
    code: str,
    registry: Registry,
    policy: CountryPresentationConfig,
    rng: random.Random,
    *,
    alias_opportunity: bool,
    language: str = "en",
    code_caption: bool = False,
    allowed_families: tuple[str, ...] | None = None,
    code_width: int | None = None,
    render_casing: RenderCasing = "preserve",
    protected_names: Sequence[str] = (),
) -> tuple[str, dict[str, Any]]:
    """Present one proven country slot, retaining its own physical line count."""
    if registry.require(text) != code:
        raise ValueError("owned country slot differs from its certified country identity")
    if any(_fold(text).casefold() == _fold(name).casefold() for name in protected_names):
        return text, {
            "family": "retained_protected_literal",
            "option": text,
            "reason": "country_word_is_also_the_protected_locality_literal",
            "eligibleFamilies": [],
            "ineligibleFamilies": [],
            "nonVaryingSlot": True,
        }
    frame = country_frame(text, code, registry)
    if frame:
        name, name_receipt = present_owned_country(
            frame["name"],
            code,
            registry,
            policy,
            rng,
            alias_opportunity=alias_opportunity,
            language=language,
            allowed_families=_NAME_FAMILIES,
            render_casing=render_casing,
        )
        printed_code, code_receipt = present_owned_country(
            frame["code"],
            code,
            registry,
            policy,
            rng,
            alias_opportunity=alias_opportunity,
            language=language,
            allowed_families=_CODE_FAMILIES,
            code_width=frame["codeWidth"],
            render_casing=render_casing,
        )
        result = name + frame["between"] + printed_code + frame["ending"]
        return result, {
            "family": "composite",
            "frame": frame,
            "name": name_receipt,
            "code": code_receipt,
        }
    families = _CODE_FAMILIES if code_caption else allowed_families
    minimum = text.count("\n") + 1
    preferred = registry.preferred_name(code)
    if (
        not alias_opportunity
        and not code_caption
        and (families is None or "name" in families)
        and len(preferred.split()) < minimum
    ):
        # This explicitly configured base behavior preserves a certified input
        # presentation when shortening it would destroy physical source lines.
        return text, {
            "family": "retained_base",
            "option": text,
            "reason": "preferred_name_cannot_preserve_owned_physical_lines",
            "eligibleFamilies": [],
            "ineligibleFamilies": ["name"],
            "nonVaryingSlot": True,
            "minimumWords": minimum,
        }
    surface, receipt = registry.choose(
        code,
        rng,
        policy,
        language=language,
        allowed_families=families,
        alias_opportunity=alias_opportunity,
        min_words=minimum,
        code_width=code_width,
    )
    surface = _case_surface(surface, text, receipt["family"], render_casing)
    result = layout_surface(text, surface)
    if result.count("\n") != text.count("\n") or registry.require(result) != code:
        raise ValueError("country presentation changed physical lines or semantic identity")
    return result, {**receipt, "minimumWords": minimum}


def _values(target: dict[str, Any]) -> dict[str, str]:
    leaves = flat(target)
    countries = {path: value for path, value in leaves.items() if path.endswith(".country")}
    addresses = {path: value for path, value in leaves.items() if path.endswith(".addressLine")}
    result = countries | addresses
    if any(not isinstance(value, str) or not value.strip() for value in result.values()):
        raise ValueError("country/address presentation inputs must be nonempty strings")
    return result


def _signature(blueprint: SamplingBlueprint) -> tuple:
    return (
        tuple(
            (r.key, r.occurrence, r.start, r.end, r.source, r.target_paths, r.curated_key)
            for r in blueprint.regions
        ),
        tuple((binding.path, binding.expression) for binding in blueprint.contract.targets),
        blueprint.document_id,
        digest(blueprint.ownership_data.get("surfaces", {})),
    )


@dataclass(frozen=True)
class _Owner:
    key: str
    occurrence: int
    source_start: int
    country_paths: tuple[str, ...]
    address_paths: tuple[str, ...]
    code: str
    postal: bool
    code_caption: bool
    ignored_couplings: tuple[str, ...]
    protected_names: tuple[str, ...]
    allowed_families: tuple[str, ...] | None = None
    code_width: int | None = None
    variable_key: str | None = None
    literal_paths: tuple[str, ...] = ()
    assembled_surface: str | None = None


@dataclass(frozen=True)
class CountryProjection:
    regions: list[tuple[RenderRegion, str]]
    target_values: dict[str, str]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class CountryRenderPlan:
    source_sha256: str
    input_values: tuple[tuple[str, str], ...]
    source_signature: tuple
    owners: tuple[_Owner, ...]
    registry: Registry
    policy: CountryPresentationConfig
    stream: DeterministicStream
    alias_opportunity: bool
    language: str
    languages: tuple[str, ...]
    language_evidence: tuple[dict[str, Any], ...]
    context_country_codes: tuple[str, ...]
    protected_localities: tuple[tuple[str, str], ...]
    protected_company_values: tuple[tuple[str, str], ...]
    protected_literal_values: tuple[tuple[str, str], ...]

    def validate_for(self, blueprint: SamplingBlueprint, target: dict[str, Any]) -> None:
        if (
            digest(blueprint.source.encode()) != self.source_sha256
            or _signature(blueprint) != self.source_signature
            or tuple(sorted(_values(target).items())) != self.input_values
            or any(flat(target).get(path) != value for path, value in self.protected_company_values)
            or any(flat(target).get(path) != value for path, value in self.protected_literal_values)
        ):
            raise ValueError("country plan differs from its pinned source, ownership or inputs")

    def apply_regions(
        self,
        rendered_regions: Sequence[tuple[RenderRegion, str]],
        *,
        variable_values: Mapping[str, str],
        render_casing: RenderCasing = "preserve",
        target_casing: TargetCasing = "uppercase",
    ) -> CountryProjection:
        """Apply exact subspan edits after canonical lexical validation/casing."""
        if target_casing not in {"uppercase", "preserve"}:
            raise ValueError("unknown country target casing")
        index = {
            (region.key, region.occurrence): (region, text) for region, text in rendered_regions
        }
        if len(index) != len(rendered_regions):
            raise ValueError("country rendering received duplicate region selectors")
        if {(item[0], item[1]) for item in self.source_signature[0]} != index.keys():
            raise ValueError("country rendering received incomplete owned regions")
        signatures = {item[:2]: item for item in self.source_signature[0]}
        for region, _ in rendered_regions:
            original = signatures[region.key, region.occurrence]
            if (
                region.key,
                region.occurrence,
                region.start,
                region.end,
                region.source,
                region.target_paths,
                region.curated_key,
            ) != original:
                raise ValueError("country rendering received a changed source owner")
        outputs, records, changes = {}, [], []
        coverage: dict[str, set[tuple[str, int]]] = defaultdict(set)
        explicit: dict[str, set[tuple[str, int]]] = defaultdict(set)
        foreign_postal_checks = []
        inputs = dict(self.input_values)
        literal_values = dict(self.protected_literal_values)
        for owner in self.owners:
            selector = owner.key, owner.occurrence
            region, text = index[selector]
            edits = []
            if owner.assembled_surface is not None:
                if _fold(text).upper() != _fold(owner.assembled_surface).upper():
                    raise ValueError("country owner differs from its certified compound expression")
            else:
                for path in owner.literal_paths:
                    if _fold(text).upper() != _fold(literal_values[path]).upper():
                        raise ValueError(
                            f"country owner differs from its joint literal target: {path}"
                        )
            spans = owned_country_spans(
                text,
                owner.code,
                self.registry,
                postal=owner.postal,
                protected_names=owner.protected_names,
            )
            for start, end, before in spans:
                # Identical repeated source presentations share a stream. A
                # mixed source alias or a different owner remains independent.
                identity = f"{owner.key}:{_fold(before).upper()}:{owner.code}"
                rng = random.Random(
                    int.from_bytes(self.stream.derive(identity).bytes(counter=0, length=16), "big")
                )
                if owner.literal_paths and owner.assembled_surface is None:
                    after, selection = (
                        before,
                        {
                            "family": "retained_literal_target",
                            "option": before,
                            "reason": "country_occurrence_jointly_owned_by_literal_target",
                            "nonVaryingSlot": True,
                            "protectedTargetPaths": list(owner.literal_paths),
                        },
                    )
                else:
                    after, selection = present_owned_country(
                        before,
                        owner.code,
                        self.registry,
                        self.policy,
                        rng,
                        alias_opportunity=self.alias_opportunity,
                        language=self.language,
                        code_caption=owner.code_caption,
                        allowed_families=owner.allowed_families,
                        code_width=owner.code_width,
                        render_casing=render_casing,
                        protected_names=owner.protected_names,
                    )
                edits.append(
                    {
                        "start": start,
                        "end": end,
                        "before": before,
                        "after": after,
                        "selection": selection,
                    }
                )
            result = text
            for edit in reversed(edits):
                if result[edit["start"] : edit["end"]] != edit["before"]:
                    raise ValueError("country subspan input changed")
                result = result[: edit["start"]] + edit["after"] + result[edit["end"] :]
            if result.count("\n") != text.count("\n"):
                raise ValueError("country projection changed owned physical line count")
            outputs[selector] = result
            if spans:
                for path in owner.country_paths:
                    coverage[path].add(selector)
                    if (
                        not owner.postal
                        and len(spans) == 1
                        and not text[: spans[0][0]].strip()
                        and not text[spans[0][1] :].strip()
                    ):
                        explicit[path].add(selector)
            if owner.postal:
                conflict = terminal_country_conflict(
                    text,
                    owner.code,
                    self.registry,
                    protected_names=owner.protected_names,
                )
                if conflict:
                    foreign_postal_checks.append(
                        {
                            "key": owner.key,
                            "occurrence": owner.occurrence,
                            "countryPaths": list(owner.country_paths),
                            "expectedCode": owner.code,
                            "terminalCountry": conflict,
                        }
                    )
            record = {
                "key": owner.key,
                "occurrence": owner.occurrence,
                "countryPaths": list(owner.country_paths),
                "addressPaths": list(owner.address_paths),
                "code": owner.code,
                "postalOwner": owner.postal,
                "codeCaption": owner.code_caption,
                "ignoredCouplingPaths": list(owner.ignored_couplings),
                "protectedNames": list(owner.protected_names),
                "before": text,
                "after": result,
                "edits": edits,
            }
            records.append(record)
            changes.extend((owner, edit) for edit in edits)
        country_paths = {path for path in inputs if path.endswith(".country")}
        if missing := country_paths - coverage.keys():
            raise ValueError(f"country target lacks an owned printed occurrence: {sorted(missing)}")
        for owner in self.owners:
            for path in owner.ignored_couplings:
                if not (coverage[path] - {(owner.key, owner.occurrence)}):
                    raise ValueError("excluded country coupling lacks a separate printed owner")
        for check in foreign_postal_checks:
            separate = {
                path: sorted(explicit[path]) for path in check["countryPaths"] if explicit[path]
            }
            if not separate:
                raise ValueError(
                    "owned postal ends with a contradictory foreign country: "
                    f"{check['key']} {check['terminalCountry']['surface']}"
                )
            check["decision"] = "retain_foreign_postal_text_with_separate_country_declaration"
            check["separateDeclarationOwners"] = separate
        projected = dict(inputs)
        for path in country_paths:
            projected[path] = self.registry.normalized(inputs[path])
        address_proof = []
        for path, value in inputs.items():
            if not path.endswith(".addressLine"):
                continue
            role = path.rsplit(".", 1)[0]
            own = [(owner, edit) for owner, edit in changes if path in owner.address_paths]
            country_value = inputs.get(role + ".country")
            if country_value is None:
                own_codes = {owner.code for owner, _ in own}
                if not own_codes:
                    continue
                if len(own_codes) != 1:
                    raise ValueError(
                        f"address without country target has conflicting owners: {path}"
                    )
                code = next(iter(own_codes))
            else:
                code = self.registry.require(country_value)
            locality = dict(self.protected_localities).get(role)
            protected = tuple(
                name for target, name in self.protected_company_values if target == role + ".name"
            ) + ((locality,) if locality else ())
            spans = owned_country_spans(
                value,
                code,
                self.registry,
                postal=True,
                protected_names=protected,
            )
            expression = next(
                (expression for target, expression in self.source_signature[1] if target == path),
                "",
            )
            variable_ranges = []
            cursor = 0
            for variable in re.findall(r"\{([^{}]+)\}", expression):
                if variable not in variable_values:
                    raise ValueError(f"address expression lacks its canonical variable: {variable}")
                match = _phrase(variable_values[variable]).search(value, cursor)
                if match is not None:
                    variable_ranges.append((match.start(), match.end(), variable))
                    cursor = match.end()
            # A country literal can sit outside the address variable while
            # remaining part of the address target expression (NAME(CODE)).
            if not own and expression:
                own = [
                    (owner, edit)
                    for owner, edit in changes
                    if role + ".country" in owner.country_paths
                ]
            replacements, split_frames = [], []
            for start, end, before in spans:
                variable = next(
                    (key for lo, hi, key in variable_ranges if lo <= start and end <= hi), None
                )
                eligible = [
                    (owner, edit)
                    for owner, edit in own
                    if variable is None or owner.variable_key == variable
                ]
                exact = [
                    (owner, edit)
                    for owner, edit in eligible
                    if _fold(edit["before"]).upper() == _fold(before).upper()
                ]
                if exact:
                    owner, edit = min(exact, key=lambda item: item[0].source_start)
                    replacements.append(
                        (
                            start,
                            end,
                            edit["after"],
                            owner.key,
                            owner.occurrence,
                            "matching_owned_country_presentation",
                        )
                    )
                elif eligible:
                    frame = country_frame(before, code, self.registry)
                    frame_pairs = []
                    if frame:
                        for left, right in pairwise(self.owners):
                            if (
                                left.allowed_families != _NAME_FAMILIES
                                or right.allowed_families != _CODE_FAMILIES
                                or left.code != code
                                or right.code != code
                                or not set(left.country_paths) & set(right.country_paths)
                            ):
                                continue
                            names = [
                                edit
                                for owner, edit in eligible
                                if owner == left
                                and _fold(edit["before"]).upper() == _fold(frame["name"]).upper()
                            ]
                            codes = [
                                edit
                                for owner, edit in eligible
                                if owner == right
                                and _fold(edit["before"]).upper() == _fold(frame["code"]).upper()
                            ]
                            if len(names) == len(codes) == 1:
                                frame_pairs.append((left, right, names[0], codes[0]))
                    if frame_pairs:
                        left, right, name_edit, code_edit = min(
                            frame_pairs, key=lambda item: item[0].source_start
                        )
                        after = (
                            name_edit["after"]
                            + frame["between"]
                            + code_edit["after"]
                            + frame["ending"]
                        )
                        replacements.append(
                            (
                                start,
                                end,
                                after,
                                left.key,
                                left.occurrence,
                                "matching_owned_split_country_frame",
                            )
                        )
                        split_frames.append(
                            {
                                "start": start,
                                "end": end,
                                "owners": [
                                    [left.key, left.occurrence],
                                    [right.key, right.occurrence],
                                ],
                                "before": before,
                                "after": after,
                                "decision": "first_complete_source_frame",
                            }
                        )
                        continue
                    alternatives = {_fold(edit["after"]).upper() for _, edit in eligible}
                    if len(alternatives) != 1:
                        raise ValueError(
                            f"address country has ambiguous owned presentations: {path}"
                        )
                    owner, edit = min(eligible, key=lambda item: item[0].source_start)
                    replacements.append(
                        (
                            start,
                            end,
                            edit["after"],
                            owner.key,
                            owner.occurrence,
                            "single_owned_country_presentation",
                        )
                    )
            result = value
            for start, end, after, _, _, _ in reversed(replacements):
                result = result[:start] + _fold(after) + result[end:]
            projected[path] = result.upper() if target_casing == "uppercase" else result
            address_proof.append(
                {
                    "path": path,
                    "before": value,
                    "after": projected[path],
                    "replacements": [list(item) for item in replacements],
                    "splitCountryFrames": split_frames,
                }
            )
        return CountryProjection(
            [
                (region, outputs.get((region.key, region.occurrence), text))
                for region, text in rendered_regions
            ],
            projected,
            {
                "version": 1,
                "sourceSha256": self.source_sha256,
                "registryManifestSha256": self.registry.manifest_sha256,
                "policy": self.policy.model_dump(mode="json"),
                "contextCountryCodes": list(self.context_country_codes),
                "protectedLocalities": dict(self.protected_localities),
                "protectedCompanyValues": dict(self.protected_company_values),
                "protectedLiteralValues": literal_values,
                "aliasOpportunity": self.alias_opportunity,
                "language": self.language,
                "eligibleLanguages": list(self.languages),
                "languageEvidence": list(self.language_evidence),
                "inputValues": inputs,
                "targetValues": projected,
                "owners": records,
                "addressProjections": address_proof,
                "foreignPostalChecks": foreign_postal_checks,
            },
        )


def prepare_country_rendering(
    blueprint: SamplingBlueprint,
    sampled_target: dict[str, Any],
    *,
    registry: Registry,
    policy: CountryPresentationConfig,
    context_country_codes: Collection[str],
    stream: DeterministicStream,
    protected_localities: Mapping[str, str] | None = None,
) -> CountryRenderPlan:
    """Pin country identities and exact owners before native byte rendering."""
    values, before = _values(sampled_target), _values(blueprint.target)
    if values.keys() != before.keys():
        raise ValueError("country presentation cannot change country/address topology")
    codes = {
        path: registry.require(value, field=path)
        for path, value in values.items()
        if path.endswith(".country")
    }
    context_codes = set(context_country_codes)
    reviewed_evidence = []
    raw = blueprint.source.encode()
    for declaration in policy.reviewed_contexts:
        if declaration.source_id != blueprint.document_id:
            continue
        registry.preferred_name(declaration.country_code)
        matches = list(re.finditer(re.escape(declaration.literal), blueprint.source))
        if not matches:
            raise ValueError("reviewed country-language context literal is absent from source")
        fixed = []
        for match in matches:
            start = len(blueprint.source[: match.start()].encode())
            end = len(blueprint.source[: match.end()].encode())
            if not any(start < region.end and region.start < end for region in blueprint.regions):
                fixed.append({"sourceByteStart": start, "sourceByteEnd": end})
        if not fixed:
            raise ValueError("reviewed country-language context literal belongs to a mutable owner")
        context_codes.add(declaration.country_code)
        reviewed_evidence.append(
            {
                "sourceId": declaration.source_id,
                "literal": declaration.literal,
                "country": declaration.country_code,
                "fixedOccurrences": fixed,
            }
        )
    context = tuple(sorted(context_codes))
    if any(registry.require(code) != code for code in context) or not set(codes.values()) <= set(
        context
    ):
        raise ValueError("country context must explicitly include every target ISO2 identity")
    protected = {}
    for role, name in (protected_localities or {}).items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("protected country-context localities must be nonempty text")
        protected[role if role.startswith("documentPatch.") else "documentPatch." + role] = name
    languages, evidence = registry.eligible_languages(context, enabled=policy.language_allowlist)
    evidence = tuple(evidence) + tuple(reviewed_evidence)
    rng = random.Random(
        int.from_bytes(stream.derive("document").bytes(counter=0, length=16), "big")
    )
    alias = rng.random() < policy.alias_probability
    language = rng.choice(languages) if alias else "en"
    binding_paths = defaultdict(set)
    for binding in blueprint.contract.targets:
        for placeholder in re.findall(r"\{([^{}]+)\}", binding.expression):
            binding_paths[placeholder].add(binding.path)
    owners = []
    all_leaves, source_leaves = flat(sampled_target), flat(blueprint.target)
    literal_values = {}
    for region in blueprint.regions:
        paths = set(region.target_paths) | binding_paths.get(region.curated_key, set())
        addresses = tuple(
            sorted(path for path in paths if path.endswith(".addressLine") and path in values)
        )
        primary = {
            path.rsplit(".", 1)[0] + ".country"
            for path in addresses
            if path.rsplit(".", 1)[0] + ".country" in codes
        }
        explicit_countries = {path for path in paths if path in codes}
        countries = explicit_countries | primary
        # An address expression may contain separately owned street, postcode
        # and reference fragments. Only a source-grounded country component
        # may borrow its party's country ownership; arbitrary generated text
        # inside another fragment is not new authority to rewrite that field.
        for path in primary - explicit_countries:
            source_code = registry.require(before[path], field=path)
            role = path.rsplit(".", 1)[0]
            source_protected = tuple(
                value
                for suffix in (".name", ".locality")
                if isinstance(value := source_leaves.get(role + suffix), str)
            )
            if not owned_country_spans(
                region.source,
                source_code,
                registry,
                postal=True,
                protected_names=source_protected,
            ):
                countries.remove(path)
        primary &= countries
        if not countries:
            continue
        primary_codes = {codes[path] for path in primary}
        choices = primary_codes or {codes[path] for path in countries}
        if len(choices) != 1:
            raise ValueError(f"country owner joins different primary identities: {region.key}")
        code = next(iter(choices))
        owned = tuple(sorted(path for path in countries if codes[path] == code))
        ignored = tuple(sorted(countries - set(owned)))
        char_start = len(raw[: region.start].decode())
        composite = bool(addresses) or any(path.endswith(".name") for path in paths)
        owner_protected = {
            protected[path.rsplit(".", 1)[0]]
            for path in owned
            if composite and path.rsplit(".", 1)[0] in protected
        }
        owner_protected.update(
            all_leaves[path.rsplit(".", 1)[0] + ".name"]
            for path in addresses
            if path.rsplit(".", 1)[0] + ".name" in all_leaves
        )
        literal_paths = tuple(
            sorted(
                path
                for path in paths
                if path not in values
                and isinstance(source_leaves.get(path), str)
                and _fold(source_leaves[path]).upper() == _fold(region.source).upper()
            )
        )
        for path in literal_paths:
            if not isinstance(all_leaves.get(path), str) or not all_leaves[path].strip():
                raise ValueError(f"joint country literal target must remain nonempty text: {path}")
            literal_values[path] = all_leaves[path]
        assembled_surface = None
        recipe = blueprint.ownership_data.get("surfaces", {}).get(region.key)
        if literal_paths and recipe is not None and "expression" in recipe:
            if set(recipe) != {"expression"} or not isinstance(recipe["expression"], str):
                raise ValueError("country compound requires one explicit expression recipe")
            expression = recipe["expression"]
            expression_paths = set(re.findall(r"\{([^{}]+)\}", expression))
            if set(literal_paths) & expression_paths and set(owned) & expression_paths:
                if expression_paths != set(owned) | set(literal_paths):
                    raise ValueError(
                        "country compound expression differs from its owned scalar paths"
                    )
                if any(
                    not isinstance(all_leaves.get(path), str) or not all_leaves[path].strip()
                    for path in expression_paths
                ):
                    raise ValueError("country compound inputs must be explicit nonempty text")
                # A scalar locality is not the complete surface of an explicitly
                # assembled locality/country cell. A whole literal expression
                # references only its full target and keeps its strict guard.
                assembled_surface = re.sub(
                    r"\{([^{}]+)\}", lambda match: all_leaves[match[1]], expression
                )
        owners.append(
            _Owner(
                region.key,
                region.occurrence,
                region.start,
                owned,
                addresses,
                code,
                bool(addresses) or region.kind == "postal",
                country_code_caption(blueprint.source, char_start),
                ignored,
                tuple(sorted(owner_protected)),
                variable_key=region.curated_key,
                literal_paths=literal_paths,
                assembled_surface=assembled_surface,
            )
        )
    # NAME(CODE) sometimes has two disjoint owners with the literal brackets
    # between them. Constrain the pieces jointly without absorbing that frame.
    regions = {(region.key, region.occurrence): region for region in blueprint.regions}
    for index, (left, right) in enumerate(pairwise(owners)):
        a, b = regions[left.key, left.occurrence], regions[right.key, right.occurrence]
        if (
            a.end <= b.start
            and set(left.country_paths) & set(right.country_paths)
            and left.code == right.code
            and not left.postal
            and not right.postal
            and not _CODE.fullmatch(_fold(a.source))
            and _CODE.fullmatch(_fold(b.source))
            and re.fullmatch(rb"\s*(?:-\s*)?\(\s*", raw[a.end : b.start])
            and re.match(rb"\s*\)", raw[b.end :])
        ):
            source_code = registry.require(a.source)
            if registry.require(b.source) != source_code:
                raise ValueError("split country frame has inconsistent source identities")
            owners[index] = replace(left, allowed_families=_NAME_FAMILIES)
            owners[index + 1] = replace(
                right,
                allowed_families=_CODE_FAMILIES,
                code_width=len(_fold(b.source).replace(".", "")),
            )
    return CountryRenderPlan(
        digest(raw),
        tuple(sorted(values.items())),
        _signature(blueprint),
        tuple(owners),
        registry,
        policy,
        stream,
        alias,
        language,
        languages,
        evidence,
        context,
        tuple(sorted(protected.items())),
        tuple(
            sorted(
                (path, value)
                for path, value in all_leaves.items()
                if path.endswith(".name")
                and (
                    path.rsplit(".", 1)[0] + ".country" in codes
                    or path.rsplit(".", 1)[0] + ".addressLine" in values
                )
                and path.startswith("documentPatch.parties.")
            )
        ),
        tuple(sorted(literal_values.items())),
    )
