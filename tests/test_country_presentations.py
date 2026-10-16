"""Production country vocabulary: identity, eligibility, pins and distributions."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.country_presentations import (
    DEFAULT_MANIFEST_SHA256,
    DEFAULT_REGISTRY_MANIFEST,
    CountryFamilyWeights,
    CountryLanguageContext,
    CountryPresentationConfig,
    Registry,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load(ROOT)


def test_pinned_complete_vocabulary(registry: Registry) -> None:
    manifest = ROOT / DEFAULT_REGISTRY_MANIFEST
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == DEFAULT_MANIFEST_SHA256
    assert len(registry.codes) == 249
    presentations = json.loads((manifest.parent / "presentations.json").read_bytes())
    assert len(presentations["presentations"]) == 3401
    for row in presentations["presentations"]:
        assert registry.resolve(row["surface"]) == row["code"]
        assert registry.find_spans(row["surface"], row["code"])
    observed = json.loads((manifest.parent / "input-aliases.json").read_bytes())
    assert len(observed["aliases"]) == 250
    for row in observed["aliases"]:
        assert registry.resolve(row["surface"]) == row["code"]
    for code in registry.codes:
        assert registry.normalized(code) == registry.preferred_name(code)
        assert registry.resolve(registry.preferred_name(code)) == code


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("  United States  ", "US"),
        ("UNITED,STATES", "US"),
        ("BELGIAN", "BE"),
        ("ARE", "AE"),
        ("A.R.E", "EG"),
        ("A.R.E.", "EG"),
        ("KOREA", "KR"),
        ("SOUTH KOREA", "KR"),
        ("NORTH KOREA", "KP"),
        ("UAE", "AE"),
        ("U.K.", "GB"),
        ("PRC", "CN"),
        ("D.R.C", "CD"),
        ("RU", "RU"),
        ("Çad", "TD"),
        ("EGITTO", "EG"),
        ("ROYAUME-UNI", "GB"),
        ("PAYS-BAS", "NL"),
        ("CONGO (DRC)", "CD"),
        ("CONGO(DRC)", "CD"),
        ("CONGO(RDC)", "CD"),
        ("MYANMAR (BURMA)", "MM"),
        ("China(CN)", "CN"),
        ("EGYPT - (EG)", "EG"),
        ("EGYPT\n(EG)", "EG"),
        ("USA-US", "US"),
        ("CHINA(CHN)", "CN"),
        ("EGYPT(E.G.)", "EG"),
    ],
)
def test_exact_aliases_and_name_code_frames(registry: Registry, value: str, code: str) -> None:
    assert registry.resolve(value) == code


@pytest.mark.parametrize(
    "value",
    [
        "EGIPT",
        "THE NETHERLANDSS",
        "ABD",
        "BK",
        "EUA",
        "R.-U.",
        "CHINA(EG)",
        "KOREA(KP)",
        "unknown",
        "",
        None,
    ],
)
def test_unknown_ambiguous_or_contradictory_is_visible(registry: Registry, value: str) -> None:
    with pytest.raises(ValueError):
        registry.resolve(value)
    with pytest.raises(ValueError, match=r"parties\.consignee\.country"):
        registry.require(value, field="parties.consignee.country")


def test_recognition_is_not_sampling(registry: Registry) -> None:
    assert registry.resolve("EGITTO") == "EG"
    assert "EGITTO" not in {row.surface for row in registry.options("EG", language="en")}
    assert "EGITTO" in {row.surface for row in registry.options("EG", language="it")}
    assert "KOREA" not in {row.surface for row in registry.options("KR")}
    assert "BELGIAN" not in {row.surface for row in registry.options("BE")}
    assert registry.is_atomic_name("CONGO(DRC)", "CD")
    assert registry.is_atomic_name("MYANMAR(BURMA)", "MM")
    assert registry.is_atomic_name("PAYS-BAS", "NL")
    assert not registry.is_atomic_name("CHINA(CN)")
    assert not registry.is_atomic_name("CONGO(DRC)", "CG")
    for language in ("en", "fr", "de", "es", "it", "pt", "tr"):
        for code in registry.codes:
            options = registry.options(code, language=language)
            assert len(options) == len({row.surface for row in options})
            assert all(
                row.surface != registry.preferred_name(code)
                for row in registry.options(code, language=language, exclude_preferred=True)
            )


def test_official_language_context_not_party_nationality(registry: Registry) -> None:
    assert registry.eligible_languages(["EG"])[0] == ("en",)
    languages, evidence = registry.eligible_languages(["EG", "IT"])
    assert languages == ("en", "it")
    assert {"country": "IT", "language": "it", "status": "official"} in evidence
    assert "fr" not in languages  # Regional/minority Italian French is not national context.
    assert registry.eligible_languages(["FR", "ES"])[0] == ("en", "fr", "es")
    with pytest.raises(ValueError):
        registry.eligible_languages(["ZZ"])
    with pytest.raises(ValueError):
        registry.eligible_languages(["IT"], enabled=("it",))


def test_document_base_and_explicit_family_conditioning(registry: Registry) -> None:
    policy = CountryPresentationConfig()
    rng = random.Random(81)
    assert policy.alias_probability == 0.30
    assert registry.choose("EG", rng, policy, alias_opportunity=False)[0] == "EGYPT"
    for _ in range(40):
        surface, receipt = registry.choose("US", rng, policy, alias_opportunity=True)
        assert surface != "UNITED STATES"
        assert registry.resolve(surface) == "US"
        assert receipt["reason"] == "document_alias_opportunity"
    surface, receipt = registry.choose(
        "EG", rng, policy, alias_opportunity=False, allowed_families=("iso3",), code_width=3
    )
    assert surface == "EGY"
    assert receipt["nonVaryingSlot"] is True
    assert registry.choose(
        "AE", rng, policy, alias_opportunity=True, allowed_families=("abbreviation",)
    )[0] in {"UAE", "U.A.E", "U.A.E."}
    with pytest.raises(ValueError, match="no eligible"):
        registry.choose(
            "EG", rng, policy, alias_opportunity=True, allowed_families=("abbreviation",)
        )
    surface, receipt = registry.choose("EG", rng, policy, alias_opportunity=True)
    assert "abbreviation" in receipt["ineligibleFamilies"]
    assert surface != "EGYPT"
    strict = CountryPresentationConfig(unavailable_family_policy="error")
    with pytest.raises(ValueError, match="unavailable"):
        registry.choose("EG", rng, strict, alias_opportunity=True)


def test_width_and_physical_line_constraints(registry: Registry) -> None:
    rng, policy = random.Random(9), CountryPresentationConfig()
    name_only = CountryPresentationConfig(
        family_weights=CountryFamilyWeights(
            name=1.0,
            iso2=0.0,
            iso3=0.0,
            abbreviation=0.0,
            dotted_code=0.0,
            localized_name=0.0,
        )
    )
    assert (
        registry.choose(
            "CN",
            rng,
            name_only,
            alias_opportunity=False,
            allowed_families=("iso3", "dotted_code"),
            code_width=3,
        )[0]
        == "CHN"
    )
    for _ in range(30):
        surface, _ = registry.choose(
            "AE",
            rng,
            policy,
            alias_opportunity=True,
            allowed_families=("iso2", "iso3", "dotted_code"),
            code_width=2,
        )
        assert len(surface.replace(".", "")) == 2
    with pytest.raises(ValueError, match="physical lines"):
        registry.choose("EG", rng, policy, alias_opportunity=False, min_words=2)
    with pytest.raises(ValueError, match="no eligible"):
        registry.choose(
            "AE", rng, policy, alias_opportunity=True, min_words=2, allowed_families=("iso2",)
        )


def test_no_alias_within_frame_is_explicit_not_a_guessed_identity(registry: Registry) -> None:
    surface, receipt = registry.choose(
        "CA",
        random.Random(3),
        CountryPresentationConfig(),
        alias_opportunity=True,
        allowed_families=("name",),
    )
    assert surface == "CANADA"
    assert receipt["nonVaryingSlot"] is True
    assert receipt["reason"] == "no_alias_within_frame"
    with pytest.raises(ValueError, match="unavailable"):
        registry.choose(
            "CA",
            random.Random(3),
            CountryPresentationConfig(unavailable_family_policy="error"),
            alias_opportunity=True,
            allowed_families=("name",),
        )


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1.0, True])
def test_invalid_weight(value: float) -> None:
    with pytest.raises(ValidationError):
        CountryFamilyWeights(name=value)


def test_invalid_configuration(registry: Registry) -> None:
    with pytest.raises(ValidationError):
        CountryFamilyWeights(**dict.fromkeys(CountryFamilyWeights.model_fields, 0.0))
    for options in (
        {"alias_probability": 1.1},
        {"alias_probability": True},
        {"language_allowlist": ("en", "xx")},
        {"language_allowlist": ("en", "en")},
        {"unknown": 1},
    ):
        with pytest.raises(ValidationError):
            CountryPresentationConfig(**options)
    with pytest.raises(ValueError, match="unsupported"):
        registry.options("EG", family="guessed")
    with pytest.raises(ValueError, match="language"):
        registry.options("EG", language="xx")
    with pytest.raises(ValueError, match="allowlist"):
        registry.choose(
            "EG",
            random.Random(),
            CountryPresentationConfig(),
            language="tr",
            alias_opportunity=True,
        )


def test_reviewed_literal_language_context_is_typed_and_immutable() -> None:
    context = CountryLanguageContext(
        source_id="doc_" + "8" * 64,
        literal="CMA CGM ITALY",
        country_code="IT",
    )
    assert CountryPresentationConfig(reviewed_contexts=(context,)).reviewed_contexts == (context,)
    with pytest.raises(ValidationError):
        CountryPresentationConfig(reviewed_contexts=(context, context))
    with pytest.raises(ValidationError):
        CountryLanguageContext(source_id="unknown", literal="CMA CGM ITALY", country_code="IT")
    with pytest.raises(ValidationError):
        CountryLanguageContext(source_id="doc_" + "8" * 64, literal=" ", country_code="IT")


def test_owned_country_spans_are_literal_and_do_not_cross_lines(registry: Registry) -> None:
    text = "BR-116 ROAD, BRAZIL 12345\nBRAZIL"
    spans = registry.find_spans(text, "BR", postal=True)
    assert [surface for _, _, surface in spans] == ["BRAZIL", "BRAZIL"]
    assert [surface for _, _, surface in registry.find_spans("A.R.E. / ARE", "AE")] == ["ARE"]
    assert [surface for _, _, surface in registry.find_spans("A.R.E. / ARE", "EG")] == ["A.R.E."]
    assert registry.find_spans("UNITED\nSTATES", "US") == []
    assert registry.find_spans("CAMBRIDGE", "BR") == []
    assert registry.find_spans("A.R.E.", "AR") == []
    assert registry.find_spans("C.H.N", "CH") == []
    assert registry.find_spans("DEMOCRATIC REPUBLIC OF CONGO", "CG") == []
    assert registry.find_spans("1 ROAD, EGITTO 12345", "EG", postal=True)[0][2] == "EGITTO"
    matches = registry.find_spans("UNITED\tSTATES 12345", "US")
    assert matches == [(0, 13, "UNITED\tSTATES")]
    for start, end, surface in registry.find_spans(text, "BR", postal=True):
        assert text[start:end] == surface


def test_terminal_named_country_is_scoped_and_excludes_code_forms(registry: Registry) -> None:
    text = "DJIBOUTI, FALKLAND ISLANDS(MALVINAS)"
    assert registry.terminal_named_country(text) == (
        10,
        len(text),
        "FALKLAND ISLANDS(MALVINAS)",
        "FK",
    )
    text = "23 ROAD, EGITTO.\n"
    assert registry.terminal_named_country(text) == (9, 15, "EGITTO", "EG")
    assert registry.terminal_named_country("23 ROAD, EG") is None
    assert registry.terminal_named_country("23 ROAD, A.R.E.") is None
    assert registry.terminal_named_country("23 ROAD, UAE") is None
    # A city/country homonym is returned literally, not inferred as an address
    # country. The caller must protect known locality spans before comparing it.
    assert registry.terminal_named_country("23 ROAD, DJIBOUTI")[3] == "DJ"
    assert registry.terminal_named_country("BR-116 ROAD") is None


def test_snapshot_and_records_are_immutable(registry: Registry) -> None:
    assert Registry.load(ROOT) is registry
    with pytest.raises(FrozenInstanceError):
        registry.path = Path("elsewhere")
    with pytest.raises(TypeError):
        registry._names["EG"] = "OTHER"
    option = registry.options("EG")[0]
    with pytest.raises(FrozenInstanceError):
        option.surface = "OTHER"
    with pytest.raises(TypeError):
        option.provenance[0]["source"] = "OTHER"
    detached = option.as_dict()
    detached["provenance"][0]["source"] = "OTHER"
    assert option.provenance[0]["source"] != "OTHER"


def test_pin_failures_are_visible(tmp_path: Path) -> None:
    path = tmp_path / "country-registry"
    shutil.copytree((ROOT / DEFAULT_REGISTRY_MANIFEST).parent, path)
    with pytest.raises(ValueError, match="manifest hash"):
        Registry(path, expected_manifest_sha256="0" * 64)
    target = path / "preferred-names.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="asset hash"):
        Registry(path)
