"""Structured OCR contracts, failure recovery and publication integrity."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image
from pypdf import PdfWriter

from document_ocr import cli
from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import canonical_json_sha256, sha256_file
from document_ocr.paddle_ocr import backend as backend_module
from document_ocr.paddle_ocr import pipeline
from document_ocr.paddle_ocr.backend import PaddleBackend, normalize_result
from document_ocr.paddle_ocr.config import PaddleConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def native_result():
    a = [[1, 2], [30, 2], [30, 12], [1, 12]]
    b = [[1, 22], [60, 22], [60, 35], [1, 35]]
    return {
        "dt_polys": [a, b],
        "rec_polys": [b],
        "rec_boxes": [[1, 22, 60, 35]],
        "rec_texts": ["SECOND LINE"],
        "rec_scores": [0.93],
    }


def test_recognition_uses_filtered_geometry_and_reports_unrecognized_detections():
    result = normalize_result(native_result(), 100, 100)
    assert result["lines"][0]["detection_index"] == 1
    assert result["lines"][0]["text"] == "SECOND LINE"
    assert result["unrecognized_detection_indices"] == [0]
    assert "raw_text" not in result


@pytest.mark.parametrize(
    "mutation",
    [
        lambda n: n.update(rec_scores=[]),
        lambda n: n.update(rec_scores=[float("nan")]),
        lambda n: n.update(rec_boxes=[[0, 0, 101, 20]]),
        lambda n: n.update(rec_polys=[[[1, 2], [4, 2], [4, 9], [1, 9]]]),
        lambda n: n.update(error="failed"),
        lambda n: n.pop("dt_polys"),
    ],
)
def test_malformed_results_fail_loudly(mutation):
    native = native_result()
    mutation(native)
    with pytest.raises(ValueError):
        normalize_result(native, 100, 100)


def test_empty_page_is_explicit_not_a_failed_page():
    native = {key: [] for key in ("dt_polys", "rec_polys", "rec_boxes", "rec_texts", "rec_scores")}
    result = normalize_result(native, 100, 100)
    assert result["lines"] == []
    assert result["detections"] == []


def config_fixture(tmp_path):
    config = load_config(ROOT / "configs/paddle_ocr.all_originals.local.yaml").model_dump()
    source = tmp_path / "source"
    source.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=150)
    writer.add_blank_page(width=200, height=150)
    with (source / "one.pdf").open("wb") as stream:
        writer.write(stream)
    shutil.copyfile(source / "one.pdf", source / "duplicate.pdf")
    config["sources"] = [{**config["sources"][0], "root": str(source)}]
    config["expected_source_documents"] = 2
    config["expected_unique_documents"] = 1
    config["exclusions"] = []
    config["output"]["root"] = str(tmp_path / "output")
    config["output"]["write_batch_rows"] = 1
    config["run"].update(run_id="test", expected_pages=2)
    config["concurrency"].update(renderer_processes=1, prefetch_pages=2)
    return PaddleConfig.model_validate(config)


def fake_provenance(config, project_root):
    value = {
        "inventory_sha256": sha256_file(pipeline.run_root(config) / "inventory.json"),
        "config": config.model_dump(mode="json"),
    }
    return {**value, "pipeline_fingerprint": canonical_json_sha256(value)}


class FakeBackend:
    calls = 0
    fail_second = False

    def __init__(self, settings):
        pass

    def close(self):
        pass

    def recognize(self, raster, artifact_root):
        type(self).calls += 1
        if type(self).fail_second and raster.stem == "000002":
            raise RuntimeError("injected recognition failure")
        native = native_result()
        path = artifact_root / "native.json"
        atomic_publish_json(path, native)
        with Image.open(raster) as image:
            width, height = image.size
        return {
            **normalize_result(native, width, height),
            "coordinate_frame": "rendered_page",
            "coordinate_image_path": str(raster),
            "coordinate_image_sha256": sha256_file(raster),
            "native_path": str(path),
            "native_sha256": sha256_file(path),
        }


@pytest.mark.parametrize("retain", [True, False])
def test_end_to_end_aliases_resume_and_artifact_corruption(tmp_path, monkeypatch, retain):
    config = config_fixture(tmp_path)
    config.output.retain_page_images = retain
    monkeypatch.setattr(pipeline, "_provenance", fake_provenance)
    FakeBackend.calls = 0
    FakeBackend.fail_second = False
    result = pipeline.run(config, ROOT, lambda _: None, backend_factory=FakeBackend)
    assert result["status"] == "complete"
    assert result["source_documents"] == 2
    assert result["unique_documents"] == 1
    assert result["successful_pages"] == 2
    assert FakeBackend.calls == 2
    root = pipeline.run_root(config)
    assert pq.read_table(root / "pages.parquet").num_rows == 2
    assert pq.read_table(root / "lines.parquet").column("text").to_pylist() == ["SECOND LINE"] * 2
    assert not list(root.rglob("*.txt"))
    assert bool(list((root / "rasters").rglob("*.png"))) == retain
    assert pipeline.run(config, ROOT, lambda _: None, backend_factory=FakeBackend) == result
    assert FakeBackend.calls == 2
    native = next((root / "native").rglob("native.json"))
    native.write_text("{}")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        pipeline.status(config)


def test_failed_page_is_not_committed_and_resume_only_retries_it(tmp_path, monkeypatch):
    config = config_fixture(tmp_path)
    monkeypatch.setattr(pipeline, "_provenance", fake_provenance)
    FakeBackend.calls = 0
    FakeBackend.fail_second = True
    with pytest.raises(pipeline.IncompletePaddleRun):
        pipeline.run(config, ROOT, lambda _: None, backend_factory=FakeBackend)
    root = pipeline.run_root(config)
    assert not (root / "manifest.json").exists()
    assert pipeline.status(config)["remaining_pages"] == 1
    assert len(list((root / "errors").glob("*.json"))) == 1
    FakeBackend.fail_second = False
    assert (
        pipeline.run(config, ROOT, lambda _: None, backend_factory=FakeBackend)["status"]
        == "complete"
    )
    assert FakeBackend.calls == 3


def test_inventory_exclusion_is_explicit_and_unknown_exclusions_are_rejected(tmp_path):
    config = config_fixture(tmp_path)
    inventory = pipeline.prepare_inventory(config)
    assert inventory["pages"] == 2
    assert pipeline.status(config)["status"] == "not_started"
    assert pipeline.status(config)["successful_pages"] == 0
    values = config.model_dump()
    values["run"]["run_id"] = "bad-exclusion"
    values["exclusions"] = [{"source_sha256": "a" * 64, "reason": "known corrupt PDF"}]
    with pytest.raises(ValueError, match="actual source hashes"):
        pipeline.prepare_inventory(PaddleConfig.model_validate(values))


def test_nested_cli_validates_without_ocr_runtime(tmp_path, capsys):
    config = config_fixture(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.model_dump(mode="json")))
    assert cli.main(["paddle", "validate-config", "--config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
    invalid = copy.deepcopy(config.model_dump(mode="json"))
    invalid["paddle"]["made_up_option"] = True
    path.write_text(json.dumps(invalid))
    assert cli.main(["paddle", "validate-config", "--config", str(path)]) != 0


@pytest.mark.parametrize("transformed", [False, True])
def test_native_word_geometry_and_coordinate_frame_are_preserved(tmp_path, transformed):
    settings = load_config(ROOT / "configs/paddle_ocr.all_originals.local.yaml").paddle
    settings.return_word_boxes = True
    if transformed:
        settings.document_orientation = settings.detection
    raster = tmp_path / "raster.png"
    Image.new("RGB", (100, 200), "white").save(raster)
    array = np.zeros((100, 200, 3) if transformed else (200, 100, 3), dtype=np.uint8)
    array[:, :, 2] = 255  # Paddle stores BGR, whereas PNG must be RGB.

    class Result(dict):
        @property
        def json(self):
            return {"res": native_result()}

    word_regions = [
        [[1, 22], [30, 22], [30, 35], [1, 35]],
        [[30, 22], [60, 22], [60, 35], [30, 35]],
    ]
    raw = Result(
        doc_preprocessor_res={"output_img": array},
        text_word=[["SECOND", "LINE"]],
        text_word_region=[[np.array(poly) for poly in word_regions]],
    )
    backend = object.__new__(PaddleBackend)
    backend.settings = settings
    backend.engine = SimpleNamespace(predict_iter=lambda _: iter([raw]))
    output = backend.recognize(raster, tmp_path / "native")
    assert output["coordinate_frame"] == ("preprocessed_page" if transformed else "rendered_page")
    assert (output["width"], output["height"]) == ((200, 100) if transformed else (100, 200))
    saved = json.loads(Path(output["native_path"]).read_text())
    assert saved["text_word"] == raw["text_word"]
    assert saved["text_word_region"] == [word_regions]
    with Image.open(output["coordinate_image_path"]) as image:
        assert image.getpixel((0, 0)) == ((255, 0, 0) if transformed else (255, 255, 255))
    del raw["text_word_region"]
    with pytest.raises(ValueError, match="word geometry requested"):
        backend.recognize(raster, tmp_path / "missing-word-geometry")


def test_gpu_request_cannot_fall_back_to_cpu(tmp_path, monkeypatch):
    settings = load_config(ROOT / "configs/paddle_ocr.all_originals.local.yaml").paddle
    settings.device = "gpu:0"
    monkeypatch.setattr(backend_module, "runtime_identity", lambda: {})
    monkeypatch.setattr(backend_module, "model_identity", lambda _: {})
    fake = SimpleNamespace(is_compiled_with_cuda=lambda: False)
    monkeypatch.setattr(
        backend_module,
        "import_module",
        lambda name: fake if name == "paddle" else SimpleNamespace(PaddleOCR=None),
    )
    with pytest.raises(ValueError, match="no CPU fallback"):
        PaddleBackend(settings)


def test_fail_fast_retains_only_successful_page_commits(tmp_path, monkeypatch):
    config = config_fixture(tmp_path)
    config.run.fail_fast = True
    monkeypatch.setattr(pipeline, "_provenance", fake_provenance)
    FakeBackend.calls = 0
    FakeBackend.fail_second = True
    with pytest.raises(RuntimeError, match="injected recognition failure"):
        pipeline.run(config, ROOT, lambda _: None, backend_factory=FakeBackend)
    assert pipeline.status(config)["successful_pages"] == 1
    assert not (pipeline.run_root(config) / "manifest.json").exists()


def test_changed_config_cannot_mix_with_completed_output(tmp_path, monkeypatch):
    config = config_fixture(tmp_path)
    monkeypatch.setattr(pipeline, "_provenance", fake_provenance)
    FakeBackend.calls = 0
    FakeBackend.fail_second = False
    pipeline.run(config, ROOT, lambda _: None, backend_factory=FakeBackend)
    config.raster.dpi += 1
    with pytest.raises(RuntimeError, match="immutable artifact conflicts"):
        pipeline.run(config, ROOT, lambda _: None, backend_factory=FakeBackend)
    assert FakeBackend.calls == 2
