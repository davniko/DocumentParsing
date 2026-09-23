from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import route_plan as plan


def test_route_preflight_collects_all_rejections_and_never_publishes_partial_run(
    tmp_path, monkeypatch
):
    samples = [{"sampleId": f"sample{i}", "sourceDocumentId": "source"} for i in range(3)]
    config = NS(
        template_run=None,
        sample_plan_run=None,
        sample_plan=NS(records=3),
        iso3166_snapshot=None,
        output_dir="out",
        run_name="test",
        seed=7,
        registry_exploration_permyriad=2500,
        maximum_candidates_per_sample=2,
        model_dump=lambda **kwargs: {},
    )
    monkeypatch.setattr(plan, "project_root_from_config", lambda _: tmp_path)
    monkeypatch.setattr(plan, "read_regular_file_bytes", lambda _: b"{}")
    monkeypatch.setattr(plan.RoutePlanConfig, "model_validate_json", lambda _: config)
    monkeypatch.setattr(plan.render, "_validate_committed_run", lambda *a: tmp_path)
    monkeypatch.setattr(plan, "_pin", lambda *a: tmp_path / "plan.jsonl")
    monkeypatch.setattr(plan.render, "_read_jsonl", lambda *a, **kw: samples)
    monkeypatch.setattr(plan.render, "_country_code_map", lambda *a: {})
    monkeypatch.setattr(plan, "load_support", lambda *a: (None, None, frozenset()))
    monkeypatch.setattr(plan.targets, "load_source", lambda *a: None)
    calls = []
    artifacts = {}

    class Stage:
        def __init__(self, **kwargs):
            self.stage_root = tmp_path

        def recover_interrupted_temporary_files(self):
            pass

        def publish_json(self, name, value):
            artifacts[name] = value

        def commit(self, **kwargs):
            pytest.fail("partial route plan was committed")

    def sample(*args, **kwargs):
        sid = kwargs["sample_id"]
        calls.append(sid)
        if sid != "sample1":
            raise ValueError("unrepresentable route")
        return NS(model_dump=lambda **kw: {"sample_id": sid})

    monkeypatch.setattr(plan, "StagedArtifactRun", Stage)
    monkeypatch.setattr(plan, "sample_projection", sample)
    with pytest.raises(ValueError, match="2 route samples require review; nothing published"):
        plan.run(tmp_path / "config.json")
    assert calls == ["sample0", "sample1", "sample2"]
    assert set(artifacts) == {"rejected-samples.json"}
    assert [r["sampleId"] for r in artifacts["rejected-samples.json"]] == ["sample0", "sample2"]
