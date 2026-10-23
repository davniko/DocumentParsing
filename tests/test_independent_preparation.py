"""Offline end-to-end preparation orchestration; provider boundary is a strict fake."""

from __future__ import annotations

import asyncio
import copy
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from document_ocr.atomic import AtomicConflictError
from document_ocr.hashing import sha256_file
from document_ocr.synthesis import independent_preparation as preparation
from document_ocr.synthesis.independent_packages import verify_native_choice_evidence
from document_ocr.synthesis.independent_relationships import verify_native_plan_evidence


@pytest.fixture
def setup_preparation(tmp_path, monkeypatch):
    state = SimpleNamespace(
        calls=[],
        failures=set(),
        fatal=set(),
        unknown=set(),
        source_holds=set(),
        physical_holds=set(),
        roles={},
        private_answers={},
        package_answers={},
        mutation=None,
        plans=[],
        mixed_contract=None,
        hardware_holds=set(),
        missing_certificates=set(),
        explicit_holds={},
        readiness_holds={},
        unsupported_cardinality=set(),
        unowned_packages=set(),
        goods_prepared=[],
        retry_modes=[],
        retry_receipts=[],
        container_counts={},
        cargo_presentations={},
    )

    def pin(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return {"path": name, "sha256": sha256_file(path)}

    class FakeRunner:
        def __init__(self, config):
            self.root = tmp_path
            self.output = tmp_path / config["output"]
            self.config = {"provider": {"model": "offline-test"}}
            self.spent = Decimal("0")

        async def call(self, stage, key, model, system, prompt):
            state.retry_modes.append((self.retry_rate_limited, self.retry_invalid_output))
            state.retry_receipts.append(self.retry_failed_receipt_sha256s)
            state.calls.append((stage, key, model, system, prompt))
            await asyncio.sleep(0)
            if state.mutation is not None:
                callback, state.mutation = state.mutation, None
                callback()
            if (stage, key) in state.failures:
                raise RuntimeError("explicit provider failure")
            if (stage, key) in state.fatal:
                raise AtomicConflictError("corrupted immutable native artifact")
            if stage == "independent-package-choice":
                compatible = None
                if "Candidate package to judge independently:" in prompt:
                    compatible = key not in state.unknown
                output = {
                    "alternatives": []
                    if key in state.unknown
                    else [{"packageType": "PACKAGE_BALE", "declaredUnitContext": None}],
                    "goodsForm": "audit-only model form must not replace goods",
                    "candidateCompatible": compatible,
                }
                output = state.package_answers.get(key, output)
            else:
                output = state.private_answers.get(
                    key,
                    {
                        "privatePackingRelation": "NONE",
                        "relatedPackageType": None,
                        "reason": "Audit-only proposal, no quantities.",
                    },
                )
            value = model.model_validate(output)
            call_key = preparation.digest(
                [stage, key, model.model_json_schema(), system, prompt, self.config["provider"]]
            )
            path = self.output / "calls" / f"{stage}-{call_key}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "stage": stage,
                        "identity": key,
                        "output": output,
                        "error": None,
                        "costUsd": "0",
                        "costStatus": "provider_reported",
                        "promptSha256": preparation.digest(prompt),
                        "schemaSha256": preparation.digest(model.model_json_schema()),
                    }
                )
            )
            return value

    class FakeSource:
        def __init__(self, root, config):
            assert config["sampling_mode"] == "independent"
            self.source_ids = config["source_ids"]
            self.variant_counts = dict.fromkeys(self.source_ids, config["variants_per_source"])
            self.capabilities = {
                sid: SimpleNamespace(family="ambient", identity_count=1, equipment_mix=None)
                for sid in self.source_ids
            }
            self.calls = FakeRunner(config)
            self.physical_support = SimpleNamespace()

        def blueprint(self, sid):
            package_rows = [{"packageQuantity": 100, "typeCategory": "PACKAGE_CARTON"}]
            if sid in state.unsupported_cardinality:
                package_rows.append(copy.deepcopy(package_rows[0]))
            return SimpleNamespace(
                document_id=sid,
                target={
                    "documentPatch": {
                        "containerInformation": [
                            {} for _ in range(state.container_counts.get(sid, 2))
                        ],
                        "goodsItemDetails": [
                            {
                                "description": "SOURCE COMMODITY MUST NOT CHOOSE GOODS",
                                "numberAndTypeOfPackages": package_rows,
                            }
                        ],
                    }
                },
                regions=[]
                if sid == "unowned" or sid in state.unowned_packages
                else [SimpleNamespace(target_paths=(preparation._TYPE_PATH,))],
            )

        def plan(self, *args):
            raise AssertionError("ordinary source planning was called")

    class FakeSampler:
        def __init__(self):
            self.phrases = {"520100": "Independent registry cotton"}
            self.hmt_states = {}

        @classmethod
        def from_config(cls, root, config):
            return cls()

        def prepare_request(self, **kwargs):
            state.goods_prepared.append(kwargs)
            assert kwargs["regime"] == "ambient" and kwargs["identity_count"] == 1
            request = {
                "sourceDocumentId": kwargs["source_document_id"],
                "variant": kwargs["variant"],
                "regime": kwargs["regime"],
                "goods": [{"hs6": "520100", "description": self.phrases["520100"]}],
                "goodsForm": self.phrases["520100"],
                "dangerousGoods": None,
                "equipmentMode": kwargs["equipment_mode"],
                "dgRequirements": {},
                "registrySelection": {"fullHs6": 1},
                "sourceCargoUsedForSelection": False,
            }
            if kwargs.get("cargo_presentation") is not None:
                request["cargoPresentation"] = kwargs["cargo_presentation"]
            return request

        def support(self):
            return {"registryHs6": 1, "sourceCargoAccess": False}

        def validate_request(self, request):
            assert request["sourceCargoUsedForSelection"] is False
            assert request["goodsForm"] == self.phrases["520100"]

    class FakeNumeric:
        @classmethod
        def from_path(cls, path, *, expected_sha256):
            assert sha256_file(path) == expected_sha256
            return cls()

        def apply_blueprint(self, bp, sid):
            if sid in state.source_holds:
                raise ValueError("source lacks complete numeric ownership")
            return bp

        def allowed_private_relations(self, sid):
            return state.roles.get(sid, ("NONE",))

        def certificate(self, sid):
            result = {"source": sid, "reviewStatus": "reviewed"}
            if sid in state.cargo_presentations:
                result["cargoPresentation"] = state.cargo_presentations[sid]
            return result

        def certificates(self):
            return {
                sid: {
                    **self.certificate(sid),
                    "holdReasons": state.explicit_holds.get(sid, []),
                    "sourceReadinessHolds": [
                        {
                            "category": "privacy",
                            "reason": reason,
                            "evidence": state.readiness_evidence,
                        }
                        for reason in state.readiness_holds.get(sid, [])
                    ],
                }
                for sid in state.config["source_ids"]
                if sid not in state.missing_certificates
            }

        def source_readiness_holds(self, sid):
            return tuple(
                SimpleNamespace(category="privacy", reason=reason)
                for reason in state.readiness_holds.get(sid, [])
            )

    class FakeIndependent:
        def __init__(self, root, config):
            self.config = config
            requests = json.loads((root / config["independent"]["requests"]["path"]).read_text())
            plans = json.loads((root / config["independent"]["private_plans"]["path"]).read_text())
            assert set(requests) == set(plans)
            assert len(requests) == sum(config["template_sampling"]["source_counts"].values())
            for item in requests.values():
                assert item["planningStage"] == "native_package_choice_accepted"
                assert item["countedUnitContext"] is None
                assert item["goodsForm"] == "Independent registry cotton"
            for key, item in requests.items():
                verify_native_choice_evidence(
                    root, item, provider={"model": "offline-test"}, hs_codes={"520100"}
                )
                verify_native_plan_evidence(
                    root,
                    item,
                    plans[key],
                    provider={"model": "offline-test"},
                    hs_codes={"520100"},
                    allowed_relations=state.roles.get(item["sourceDocumentId"], ("NONE",)),
                    source_certificate_sha256=preparation.digest(
                        FakeNumeric().certificate(item["sourceDocumentId"])
                    ),
                )

        def plan(self, sid, variant):
            state.plans.append((sid, variant))
            if f"{sid}:{variant}" in state.physical_holds:
                raise ValueError("explicit physical incompatibility")

    monkeypatch.setattr(preparation, "_SourceOwnershipCampaign", FakeSource)
    monkeypatch.setattr(preparation, "IndependentSampler", FakeSampler)
    monkeypatch.setattr(preparation, "IndependentNumericCertificates", FakeNumeric)
    monkeypatch.setattr(preparation, "IndependentCampaign", FakeIndependent)

    def check_hardware(bp, support, capability, regime):
        if bp.document_id in state.hardware_holds:
            raise ValueError("source equipment has no supported regime domain")
        return None, [], None

    monkeypatch.setattr(preparation, "source_equipment_readiness", check_hardware)
    monkeypatch.setattr(preparation, "validate_identifier_privacy", lambda _: None)
    monkeypatch.setattr(preparation, "validate_private_person_coverage", lambda _: None)
    monkeypatch.setattr(
        preparation, "load_independent_geography_certificates", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        preparation,
        "IndependentGeographyGate",
        lambda _: SimpleNamespace(bind_blueprint=lambda *args: None),
    )
    monkeypatch.setattr(
        preparation,
        "compile_equipment_inventory",
        lambda *a, **kw: (
            state.mixed_contract or SimpleNamespace(row_count=2, uniform_aggregate=None)
        ),
    )
    registry = {
        name: pin(f"registry/{name}.json", {})
        for name in ("hsManifest", "hsMetadata", "hsReport", "phrases", "dgHmt", "dgEcics", "flash")
    }
    state.readiness_evidence = pin("readiness-evidence.json", {"finding": "test source owner"})
    config = {
        "sampling_mode": "independent",
        "source_ids": ["a", "b"],
        "capabilities": {"a": {}, "b": {}},
        "variants_per_source": 3,
        "output": "output",
        "seed": 41,
        "dataset": "dataset",
        "ownership": "ownership.json",
        "auxiliary": "auxiliary.json",
        "task_constraints": "task.json",
        "source_contracts": "cases",
        "historical_catalog": "cases",
        "sampling": {},
        "independent": {
            "registry": registry,
            "numeric_certificates": pin("numeric.json", {}),
            "wording_certificates": pin("wording.json", {"version": 1, "sources": {}}),
            "geography_certificates": pin("geography.json", {}),
            "locality_owner_reviews": pin("locality.json", {}),
            "distinct_actor_reviews": pin("actors.json", {}),
        },
    }
    for path in (
        "ownership.json",
        "auxiliary.json",
        "task.json",
        "dataset/train.jsonl",
        "dataset/validation.jsonl",
        "cases/a/contract.json",
        "cases/a/template.json",
        "cases/b/contract.json",
        "cases/b/template.json",
    ):
        pin(path, {})
    state.config, state.root = config, tmp_path
    state.run = lambda **kw: asyncio.run(
        preparation.prepare_campaign(
            tmp_path,
            copy.deepcopy(config),
            package_targets={"PACKAGE_BALE": 6},
            **({"unit_profile_source_ids": (), **kw}),
        )
    )
    return state


def test_new_preparation_defaults_to_native_profiles_for_all_selected_sources():
    import inspect

    assert (
        inspect.signature(preparation.prepare_campaign)
        .parameters["unit_profile_source_ids"]
        .default
        is None
    )


def test_three_variants_keep_inventory_identity_and_unique_exclusion_set(setup_preparation):
    state = setup_preparation
    result = state.run()
    assert result["readySlots"] == 6
    for key, row in result["inventory"].items():
        assert row["variant"] == int(key.rsplit(":", 1)[1])
    for request in state.goods_prepared:
        excluded = request["excluded_identifiers"]
        assert len(excluded) == len(set(excluded))


def test_reviewed_cargo_domain_reaches_sampler_native_schema_and_frozen_request(setup_preparation):
    state = setup_preparation
    domain = {"hsHeadings": ["5201"], "excludedHs6": [], "packageTypes": ["PACKAGE_BALE"]}
    state.cargo_presentations["a"] = domain
    result = state.run()
    assert result["readySources"] == 2
    requests = json.loads(
        (state.root / result["config"]["independent"]["requests"]["path"]).read_text()
    )
    for variant in range(1, 4):
        request = requests[f"a:{variant}"]
        assert request["cargoPresentation"] == domain
        assert request["nativeChoiceEvidence"]["request"]["allowedPackageTypes"] == ["PACKAGE_BALE"]
        assert (
            "allowedPackageTypes" not in requests[f"b:{variant}"]["nativeChoiceEvidence"]["request"]
        )
    scoped_calls = [
        c for c in state.calls if c[0] == "independent-package-choice" and c[1].startswith("a:")
    ]
    assert len(scoped_calls) == 3
    assert all(
        c[2].model_json_schema()["properties"]["alternatives"]["maxItems"] == 1
        for c in scoped_calls
    )


def test_complete_goods_first_pipeline_is_runnable_and_keeps_final_counts_zero(setup_preparation):
    state = setup_preparation
    before = copy.deepcopy(state.config)
    result = state.run()
    assert result["readySources"] == 2 and result["readySlots"] == 6
    assert result["plannedSlots"] == 6 and result["counts"] == {"ready": 6}
    assert result["readyReservations"] == {"PACKAGE_BALE": 6}
    assert result["acceptedFinalCounts"] == {}
    assert result["coverage"]["PACKAGE_BALE"]["shortfall"] == 6
    assert len(state.calls) == 6
    assert all("SOURCE COMMODITY" not in row[4] for row in state.calls)
    assert state.config == before
    assert len(state.plans) == 6
    assert (state.root / result["preparedConfig"]["path"]).exists()


def test_mixed_native_semantics_reach_verified_production_plan(setup_preparation):
    state = setup_preparation
    state.roles["a"] = ("MIXED",)
    for variant in (1, 2, 3):
        key = f"a:{variant}"
        state.package_answers[key] = {
            "alternatives": [{"packageType": "PACKAGE_PACKAGE", "declaredUnitContext": None}],
            "goodsForm": "Audit-only",
            "candidateCompatible": True,
        }
        state.private_answers[key] = {
            "unknown": False,
            "supportPackageType": "PACKAGE_PALLET",
            "relatedPackageType": "PACKAGE_BALE",
            "reason": "Bales on pallets plus loose bales",
        }
    result = state.run()
    assert result["readySlots"] == 6
    calls = [row for row in state.calls if row[0] == "independent-mixed-accounting"]
    assert len(calls) == 3
    payload = json.loads((state.root / result["preparedConfig"]["path"]).read_text())
    plan_path = state.root / payload["independent"]["private_plans"]["path"]
    plans = json.loads(plan_path.read_text())
    assert plans["a:1"]["origin"] == "native_mixed_accounting"
    assert plans["a:1"]["privatePackingRelation"] == "MIXED"
    assert "reason" not in plans["a:1"]
    plans["a:1"]["supportPackageType"] = "PACKAGE_RACK"
    frozen = json.loads((state.root / payload["independent"]["requests"]["path"]).read_text())
    with pytest.raises(ValueError, match="differs from received native types"):
        verify_native_plan_evidence(
            state.root,
            frozen["a:1"],
            plans["a:1"],
            provider={"model": "offline-test"},
            hs_codes={"520100"},
            allowed_relations=("MIXED",),
            source_certificate_sha256="unused",
        )


def test_uniform_equipment_owner_failure_is_preflight_not_paid_failure(
    setup_preparation, monkeypatch
):
    state = setup_preparation
    original = preparation._SourceOwnershipCampaign.blueprint

    def blueprint(self, sid):
        value = original(self, sid)
        if sid == "a":
            value.target["documentPatch"]["containerInformation"] = [
                {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"}
                for _ in range(2)
            ]
        return value

    def missing_owner(*args, **kwargs):
        raise ValueError("each equipment identifier needs direct printed type ownership")

    monkeypatch.setattr(preparation._SourceOwnershipCampaign, "blueprint", blueprint)
    monkeypatch.setattr(preparation, "rebase_equipment_inventory_headers", lambda bp: bp)
    monkeypatch.setattr(preparation, "compile_equipment_inventory", missing_owner)
    result = state.run()
    assert result["plannedSlots"] == 6 and result["readySlots"] == 3
    assert all(row[1].startswith("b:") for row in state.calls)
    assert all(row["source_document_id"] == "b" for row in state.goods_prepared)
    assert set(result["config"]["independent"]) == preparation._AUTHORITIES | {
        "requests",
        "private_plans",
    }


@pytest.mark.parametrize("limited,invalid", [(False, False), (True, False), (False, True)])
def test_explicit_preparation_retry_permissions_reach_actual_runner(
    setup_preparation, limited, invalid
):
    state = setup_preparation
    result = state.run(retry_rate_limited=limited, retry_invalid_output=invalid)
    assert result["readySlots"] == 6
    assert state.retry_modes == [(limited, invalid)] * 6


@pytest.mark.parametrize("value", [1, "true", None])
def test_preparation_refuses_non_boolean_retry_permissions(setup_preparation, value):
    with pytest.raises(ValueError, match="explicit booleans"):
        setup_preparation.run(retry_invalid_output=value)
    assert not setup_preparation.calls


def test_preparation_forwards_exact_failed_receipt_retry_permissions(setup_preparation):
    state = setup_preparation
    hashes = ["a" * 64, "b" * 64]
    assert state.run(retry_failed_receipt_sha256s=hashes)["readySlots"] == 6
    assert state.retry_receipts == [frozenset(hashes)] * 6


@pytest.mark.parametrize("value", ["not-a-hash", "A" * 64, "a" * 63, 1])
def test_preparation_refuses_unpinned_failed_retry_permissions(setup_preparation, value):
    with pytest.raises(ValueError, match="exact SHA-256"):
        setup_preparation.run(retry_failed_receipt_sha256s=[value])
    assert not setup_preparation.calls


def test_inventory_compiles_all_slots_without_native_calls_goods_or_physical_draws(
    setup_preparation,
):
    state = setup_preparation
    before = {
        str(path.relative_to(state.root)): path.read_bytes()
        for path in state.root.rglob("*")
        if path.is_file()
    }
    result = preparation.inventory_sources(state.root, state.config)
    assert result["selectedSources"] == 2 and result["plannedSlots"] == 6
    assert result["readySources"] == 2 and result["readySlots"] == 6
    assert result["counts"] == {"source_prerequisites_pass": 2}
    assert result["inventory"]["a"]["variants"] == 3
    assert result["nativeCalls"] == result["cargoDraws"] == 0
    assert not state.calls and not state.goods_prepared and not state.plans
    assert {
        str(path.relative_to(state.root)): path.read_bytes()
        for path in state.root.rglob("*")
        if path.is_file()
    } == before


def test_inventory_distinguishes_missing_certificate_from_explicit_reviewed_hold(
    setup_preparation,
):
    state = setup_preparation
    state.missing_certificates.add("a")
    state.explicit_holds["b"] = ["Complete article-frame ownership requires review"]
    result = preparation.inventory_sources(state.root, state.config)
    assert result["plannedSlots"] == 6 and result["readySlots"] == 0
    assert result["inventory"]["a"]["category"] == "missing_numeric_certificate"
    assert result["inventory"]["b"]["category"] == "explicit_numeric_hold"
    assert "article-frame" in result["inventory"]["b"]["reason"]
    prepared = state.run()
    for sid in ("a", "b"):
        assert (
            prepared["inventory"][sid + ":1"]["readinessCategory"]
            == (result["inventory"][sid]["category"])
        )
    assert not state.calls and not state.goods_prepared


def test_nonnumeric_readiness_hold_is_distinct_and_blocks_before_any_native_call(
    setup_preparation,
):
    state = setup_preparation
    state.readiness_holds["a"] = ["Unowned private shipment-appointed agency"]
    result = preparation.inventory_sources(state.root, state.config)
    assert result["inventory"]["a"]["category"] == "source_readiness_hold"
    assert result["plannedSlots"] == 6 and result["readySlots"] == 3
    assert not state.calls
    prepared = state.run()
    assert prepared["inventory"]["a:1"]["readinessCategory"] == "source_readiness_hold"
    assert prepared["config"]["source_ids"] == ["b"]
    assert all(call[1].startswith("b:") for call in state.calls)


def test_nonnumeric_hold_evidence_drift_is_fatal_not_a_quiet_hold(setup_preparation):
    state = setup_preparation
    state.readiness_holds["a"] = ["Unowned private agency"]
    (state.root / "readiness-evidence.json").write_text("corrupted evidence")
    with pytest.raises(ValueError, match="pin mismatch"):
        preparation.inventory_sources(state.root, state.config)
    with pytest.raises(ValueError, match="pin mismatch"):
        state.run()
    assert not state.calls and not state.goods_prepared


def test_nonnumeric_hold_evidence_is_in_the_native_stage_pin_barrier(setup_preparation):
    state = setup_preparation
    state.readiness_holds["a"] = ["Unowned private agency"]
    state.mutation = lambda: (state.root / "readiness-evidence.json").write_text("drift")
    with pytest.raises(ValueError, match="authority changed"):
        state.run()
    assert len(state.calls) == 3 and all(call[1].startswith("b:") for call in state.calls)


@pytest.mark.parametrize(
    ("state_key", "expected"),
    [
        ("hardware_holds", "source_compile_hold"),
        ("unsupported_cardinality", "unsupported_cardinality"),
        ("unowned_packages", "unowned_package_declaration"),
    ],
)
def test_inventory_and_preparation_use_identical_source_prerequisites(
    setup_preparation, state_key, expected
):
    state = setup_preparation
    getattr(state, state_key).add("a")
    result = preparation.inventory_sources(state.root, state.config)
    assert result["inventory"]["a"]["category"] == expected
    assert result["plannedSlots"] == 6 and result["readySlots"] == 3
    assert not state.calls and not state.goods_prepared
    prepared = state.run()
    assert prepared["inventory"]["a:1"]["readinessCategory"] == expected
    assert prepared["inventory"]["a:1"]["reason"] == result["inventory"]["a"]["reason"]


def test_inventory_and_preparation_validate_privacy_ownership_before_native(
    setup_preparation, monkeypatch
):
    state = setup_preparation

    def privacy(bp):
        if bp.document_id == "a":
            raise ValueError("Private person lacks a generated owner")

    monkeypatch.setattr(preparation, "validate_private_person_coverage", privacy)
    result = preparation.inventory_sources(state.root, state.config)
    assert "Private person" in result["inventory"]["a"]["reason"]
    assert not state.calls and not state.goods_prepared
    prepared = state.run()
    assert result["inventory"]["a"]["reason"] == prepared["inventory"]["a:1"]["reason"]


@pytest.mark.parametrize("kind", ["hash", "malformed_pin", "malformed_reviews"])
def test_inventory_global_authority_corruption_is_fatal_not_a_source_hold(setup_preparation, kind):
    state = setup_preparation
    if kind == "hash":
        (state.root / "locality.json").write_text("corrupt")
    elif kind == "malformed_pin":
        state.config["independent"]["distinct_actor_reviews"]["sha256"] = "wrong"
    else:
        path = state.root / "locality.json"
        path.write_text(json.dumps({"a": []}))
        state.config["independent"]["locality_owner_reviews"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError):
        preparation.inventory_sources(state.root, state.config)
    assert not state.calls and not state.goods_prepared
    with pytest.raises(ValueError):
        state.run()
    assert not state.calls


@pytest.mark.parametrize("output", [".", "../outside"])
def test_inventory_and_preparation_reject_broad_or_external_output(setup_preparation, output):
    state = setup_preparation
    state.config["output"] = output
    with pytest.raises(ValueError, match="specific directory inside"):
        preparation.inventory_sources(state.root, state.config)
    with pytest.raises(ValueError, match="specific directory inside"):
        state.run()
    assert not state.calls


def test_native_failure_awaits_all_and_blocks_successful_siblings_without_replacement(
    setup_preparation,
):
    state = setup_preparation
    state.failures.add(("independent-package-choice", "a:2"))
    result = state.run()
    assert len(state.calls) == 6
    assert result["readySources"] == 1 and result["readySlots"] == 3
    assert result["inventory"]["a:2"]["status"] == "held_native_choice"
    assert result["inventory"]["a:1"]["status"] == "blocked_by_source_group"
    assert result["inventory"]["a:3"]["status"] == "blocked_by_source_group"
    assert result["config"]["source_ids"] == ["b"]
    assert result["config"]["template_sampling"] == {"samples": 3, "source_counts": {"b": 3}}
    assert set(result["inventory"]) == {f"{sid}:{v}" for sid in ("a", "b") for v in (1, 2, 3)}


def test_unknown_never_becomes_a_default_package(setup_preparation):
    state = setup_preparation
    state.unknown.add("a:1")
    result = state.run()
    assert result["inventory"]["a:1"]["status"] == "held_unknown_package"
    assert result["readySlots"] == 3


def test_local_integrity_failure_is_fatal_after_every_native_attempt_finishes(setup_preparation):
    state = setup_preparation
    state.fatal.add(("independent-package-choice", "a:2"))
    with pytest.raises(
        RuntimeError, match=r"all scheduled requests completed.*AtomicConflictError"
    ):
        state.run()
    assert len(state.calls) == 6


def test_missing_source_certificate_blocks_before_any_call_for_that_source(setup_preparation):
    state = setup_preparation
    state.source_holds.add("a")
    result = state.run()
    assert len(state.calls) == 3 and all(row[1].startswith("b:") for row in state.calls)
    assert result["counts"] == {"held_source": 3, "ready": 3}


def test_wrong_source_hardware_is_held_before_native_not_rescued_by_new_goods(setup_preparation):
    state = setup_preparation
    state.hardware_holds.add("a")
    result = state.run()
    assert result["config"]["source_ids"] == ["b"]
    assert all(not item[1].startswith("a:") for item in state.calls)
    assert "regime domain" in result["inventory"]["a:1"]["reason"]


def test_all_sources_held_produce_full_inventory_but_no_fake_runnable_config(setup_preparation):
    state = setup_preparation
    state.source_holds.update({"a", "b"})
    result = state.run()
    assert state.calls == [] and result["config"] is None
    assert result["preparedConfig"] is None and result["plannedSlots"] == 6
    assert result["counts"] == {"held_source": 6}


def test_private_typed_calls_are_second_stage_and_never_use_reason_as_facts(setup_preparation):
    state = setup_preparation
    state.roles["a"] = ("NONE", "CONTAINS")
    state.private_answers["a:1"] = {
        "privatePackingRelation": "CONTAINS",
        "relatedPackageType": "PACKAGE_BAG_PLASTIC",
        "reason": "One bag per bale: audit-only, not a host count.",
    }
    result = state.run()
    assert len(state.calls) == 9
    assert all(row[0] == "independent-package-choice" for row in state.calls[:6])
    pin = result["config"]["independent"]["private_plans"]
    plans = json.loads((state.root / pin["path"]).read_text())
    assert plans["a:1"]["privatePackingRelation"] == "CONTAINS"
    assert "reason" not in plans["a:1"]
    assert "One bag per bale" in plans["a:1"]["nativePlanEvidence"]["output"]["reason"]


def test_private_unknown_blocks_only_its_source_group(setup_preparation):
    state = setup_preparation
    state.roles["a"] = ("NONE", "CONTAINS")
    state.private_answers["a:1"] = {
        "privatePackingRelation": "UNKNOWN",
        "relatedPackageType": None,
        "reason": "No workable plan",
    }
    result = state.run()
    assert result["inventory"]["a:1"]["status"] == "held_unknown_private_plan"
    assert result["config"]["source_ids"] == ["b"]


def test_actual_physical_failure_prevents_emitting_incomplete_source_group(setup_preparation):
    state = setup_preparation
    state.physical_holds.add("a:3")
    result = state.run()
    assert result["config"]["source_ids"] == ["b"]
    assert result["inventory"]["a:3"]["status"] == "held_physical_plan"
    assert result["inventory"]["a:1"]["status"] == "blocked_by_source_group"
    assert len(state.plans) == 6


def test_authority_drift_during_native_calls_fails_closed(setup_preparation):
    state = setup_preparation
    state.mutation = lambda: (state.root / "ownership.json").write_text("changed")
    with pytest.raises(ValueError, match="authority changed"):
        state.run()
    assert len(state.calls) == 6


def test_mixed_mode_is_deliberate_and_checked_against_source_ownership(setup_preparation):
    state = setup_preparation
    result = state.run(mixed_source_ids=["a"], mixed_probability=1)
    pin = result["config"]["independent"]["requests"]
    requests = json.loads((state.root / pin["path"]).read_text())
    assert {r["equipmentMode"] for key, r in requests.items() if key.startswith("a:")} == {"mixed"}
    assert {r["equipmentMode"] for key, r in requests.items() if key.startswith("b:")} == {
        "uniform"
    }
    for key, row in requests.items():
        slots = row["equipmentSlots"]
        assert len(slots) == 2
        assert (len({tuple(sorted(s.items())) for s in slots}) == 2) == key.startswith("a:")
        assert row["nativeChoiceEvidence"]["request"]["equipment"] == (
            preparation.equipment_choice_brief(slots)
        )


def test_uncontainerized_source_does_not_invent_equipment_in_native_brief(setup_preparation):
    state = setup_preparation
    state.container_counts["a"] = 0
    result = state.run()
    request_path = state.root / result["config"]["independent"]["requests"]["path"]
    rows = json.loads(request_path.read_text())
    for key, row in rows.items():
        if key.startswith("a:"):
            assert row["equipmentSlots"] == []
            assert (
                "no printed carrying containers"
                in (row["nativeChoiceEvidence"]["request"]["equipment"])
            )


def test_unowned_mixed_presentation_is_held_not_silently_uniform(setup_preparation):
    state = setup_preparation
    state.mixed_contract = SimpleNamespace(row_count=2, uniform_aggregate=object())
    result = state.run(mixed_source_ids=["a"], mixed_probability=1)
    assert result["config"]["source_ids"] == ["b"]
    assert all(not call[1].startswith("a:") for call in state.calls)


@pytest.mark.parametrize("probability", [-1, 1.01, float("nan"), True])
def test_invalid_mixed_policy_fails_before_calls(setup_preparation, probability):
    state = setup_preparation
    with pytest.raises(ValueError, match="finite probability"):
        state.run(mixed_probability=probability)
    assert not state.calls


def test_immutable_final_maps_conflict_instead_of_overwriting(setup_preparation):
    state = setup_preparation
    result = state.run()
    path = state.root / result["config"]["independent"]["requests"]["path"]
    path.write_text("corrupt")
    with pytest.raises(RuntimeError, match="immutable artifact conflicts"):
        state.run()


def test_source_only_loader_never_exposes_observed_sampling():
    loader = object.__new__(preparation._SourceOwnershipCampaign)
    with pytest.raises(RuntimeError, match="never sample observed cargo"):
        loader.plan("source", 1)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "changed_output",
        "changed_category",
        "context",
        "receipt_pin",
        "absolute_path",
        "duplicate_receipt",
    ],
)
def test_native_evidence_cannot_be_replaced_by_planning_stage_or_forged_metadata(
    setup_preparation, mutation
):
    state = setup_preparation
    result = state.run()
    pin = result["config"]["independent"]["requests"]
    request = json.loads((state.root / pin["path"]).read_text())["a:1"]
    if mutation == "missing":
        del request["nativeChoiceEvidence"]
    elif mutation == "changed_output":
        request["nativeChoiceEvidence"]["output"]["goodsForm"] = "forged native output"
    elif mutation == "changed_category":
        request["selectedCategory"] = "PACKAGE_SET"
    elif mutation == "context":
        request["countedUnitContext"] = "vehicle"
    elif mutation == "receipt_pin":
        request["nativeChoiceEvidence"]["callReceipts"][0]["sha256"] = "0" * 64
    elif mutation == "absolute_path":
        request["nativeChoiceEvidence"]["packet"]["path"] = str(state.root / "packet.json")
    else:
        request["nativeChoiceEvidence"]["callReceipts"] *= 2
    with pytest.raises(ValueError):
        verify_native_choice_evidence(
            state.root, request, provider={"model": "offline-test"}, hs_codes={"520100"}
        )


def test_native_evidence_requires_the_actual_successful_output_not_just_an_input_packet(
    setup_preparation,
):
    state = setup_preparation
    result = state.run()
    pin = result["config"]["independent"]["requests"]
    request = json.loads((state.root / pin["path"]).read_text())["a:1"]
    request["nativeChoiceEvidence"]["callReceipts"] = []
    with pytest.raises(ValueError, match="no canonical Runner receipts"):
        verify_native_choice_evidence(
            state.root, request, provider={"model": "offline-test"}, hs_codes={"520100"}
        )


def test_reviewed_none_requires_exact_source_certificate_not_a_free_default(setup_preparation):
    state = setup_preparation
    result = state.run()
    requests_pin = result["config"]["independent"]["requests"]
    request = json.loads((state.root / requests_pin["path"]).read_text())["a:1"]
    plans_pin = result["config"]["independent"]["private_plans"]
    plan = json.loads((state.root / plans_pin["path"]).read_text())["a:1"]
    with pytest.raises(ValueError, match="source certificate"):
        verify_native_plan_evidence(
            state.root,
            request,
            plan,
            provider={"model": "offline-test"},
            hs_codes={"520100"},
            allowed_relations=("NONE",),
            source_certificate_sha256="0" * 64,
        )
