import json
from copy import deepcopy
from pathlib import Path
from random import Random

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.semantic_v3.config import FrozenArtifactInput
from document_ocr.semantic_v3.transform import CategoryRegistry, _load_frozen_model
from document_ocr.synthesis.package_aliases import (
    select_package_alias,
    validate_package_alias_registry,
)
from document_ocr.synthesis.package_registry import PackageRegistryPayload, load_package_registry

ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT / "artifacts/mpci-ai-schema/categorical-registry-followup/package-category-registry.json"
)
VOCABULARY = ROOT / "artifacts/sampling-registries/package-extraction-vocabulary-v2.json"
PIN = "4392e46f19ef4cbc6138c6039c3cafaf4c36edacc5bacd468baec635820058c0"


def value():
    return json.loads(VOCABULARY.read_bytes())


def test_pinned_407_has_exact_immutable_405_plus_two_explicit_semantic_extensions():
    before = json.loads(BASE.read_bytes())
    after = value()
    assert sha256_file(BASE) == "2476deb46c3cabf2efe9a00bbb9ad1bf6b527e4adae0e4872d8a63807bbd374f"
    assert after["entries"][:405] == before["entries"]
    registry = load_package_registry(VOCABULARY, expected_sha256=PIN, expected_entries=407)
    assert registry.application_code("PACKAGE_UNIT") == "UN"
    for token in ("PACKAGE_SET", "PACKAGE_BALE"):
        assert registry.entry(token).applicationCode is None
        with pytest.raises(ValueError, match="no exact application-code mapping"):
            registry.application_code(token)
    assert select_package_alias("PACKAGE_SET", 1, "SETS", mode="source", rng=Random(1)) == "SET"
    assert (
        select_package_alias("PACKAGE_BALE", 2, "BALE", mode="registry", rng=Random(1)) == "BALES"
    )
    validate_package_alias_registry(
        {"PACKAGE_SET", "PACKAGE_BALE"}, {"PACKAGE_SET": None, "PACKAGE_BALE": None}
    )
    with pytest.raises(ValueError, match="codes differ"):
        validate_package_alias_registry({"PACKAGE_SET"}, {"PACKAGE_SET": "UN"})


def test_semantic_category_loader_verifies_same_source_and_rejects_unmapped_projection():
    registry, pin = _load_frozen_model(
        FrozenArtifactInput(path=str(VOCABULARY), sha256=PIN), CategoryRegistry
    )
    assert pin == PIN and len(registry.entries) == 407
    with pytest.raises(ValueError, match="no exact application-code mapping"):
        registry.application_code("PACKAGE_BALE")


@pytest.mark.parametrize("model", [PackageRegistryPayload, CategoryRegistry])
@pytest.mark.parametrize(
    "mutation",
    [
        "v1",
        "missing_reason",
        "missing_provenance",
        "extra_review",
        "fake_set_code",
        "container",
        "missing_source",
    ],
)
def test_unreviewed_or_fake_wire_semantics_fail_closed(model, mutation):
    data = value()
    if mutation == "v1":
        data["schemaVersion"] = 1
    elif mutation == "missing_reason":
        del data["semanticExtensions"]["PACKAGE_SET"]["noExactApplicationCodeReason"]
    elif mutation == "missing_provenance":
        data["semanticExtensions"]["PACKAGE_BALE"]["reviewProvenance"] = "  "
    elif mutation == "extra_review":
        data["semanticExtensions"]["PACKAGE_UNIT"] = deepcopy(
            data["semanticExtensions"]["PACKAGE_SET"]
        )
    elif mutation == "fake_set_code":
        data["entries"][-2]["applicationCode"] = "UN"
    elif mutation == "container":
        data["registryKind"] = "container"
    else:
        del data["applicationRegistry"]
    with pytest.raises(ValueError):
        model.model_validate_json(json.dumps(data), strict=True)


def test_changed_application_entries_or_source_pin_fail_at_loader_boundary(tmp_path):
    original = value()
    original["applicationRegistry"]["path"] = "base.json"
    (tmp_path / "base.json").write_bytes(BASE.read_bytes())
    for field in ("entry", "pin"):
        data = deepcopy(original)
        if field == "entry":
            data["entries"][0]["displayName"] += " ALTERED"
        else:
            data["applicationRegistry"]["sha256"] = "0" * 64
        path = tmp_path / "vocabulary.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="application"):
            load_package_registry(path, expected_sha256=sha256_file(path), expected_entries=407)
