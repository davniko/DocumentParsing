import pytest

from document_ocr.synthesis.paddle_render import project_printed_postal_country


class RegistryFixture:
    def preferred_name(self, code):
        assert code == "RU"
        return "RUSSIA"

    def find_spans(self, text, code, *, postal):
        assert code == "RU" and postal
        return [1] if "RUSSIAN FEDERATION" in text else []

    def resolve(self, alias):
        return {"RUSSIAN FEDERATION": "RU", "BELGIUM": "BE"}[alias]


def run(monkeypatch, country, address="ADDR RUSSIAN FEDERATION"):
    from document_ocr.synthesis.country_presentations import Registry
    monkeypatch.setattr(Registry, "load", lambda *a, **kw: RegistryFixture())
    target = {"party": {"addressLine": address, "country": country}}
    project_printed_postal_country(target, {"scenario": {"origin": {"country_code": "RU"}}},
        {"manifestPath": "pinned.json", "sha256": "pinned",
         "addressTarget": "/party/addressLine", "codePath": "/origin/country_code",
         "target": "/party/country"}, None)
    return target


def test_same_identity_alias_normalizes_only_with_owned_postal_evidence(monkeypatch):
    assert run(monkeypatch, "RUSSIAN FEDERATION")["party"]["country"] == "RUSSIA"
    assert run(monkeypatch, "RUSSIA")["party"]["country"] == "RUSSIA"
    assert run(monkeypatch, "RUSSIAN FEDERATION", "ADDR")["party"]["country"] == "RUSSIAN FEDERATION"


def test_different_or_unknown_existing_identity_still_fails(monkeypatch):
    with pytest.raises(ValueError, match="conflicts"):
        run(monkeypatch, "BELGIUM")
    with pytest.raises(KeyError):
        run(monkeypatch, "UNKNOWN")
