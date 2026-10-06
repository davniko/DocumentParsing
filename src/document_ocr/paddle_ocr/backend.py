"""Local PaddleOCR adapter with explicit geometry and lossless native artifacts."""

from __future__ import annotations

import importlib.metadata
import io
import json
import os
from collections import defaultdict, deque
from importlib import import_module
from pathlib import Path
from typing import Any

from PIL import Image

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import canonical_json_sha256, sha256_file
from document_ocr.paddle_ocr.config import PaddleSettings

RUNTIME_VERSIONS = {"paddleocr": "3.7.0", "paddlex": "3.7.2", "paddle": "3.3.1"}


def model_directory(settings: PaddleSettings, role: str) -> Path:
    spec = settings.models()[role]
    return Path(settings.models_directory) / spec.repository / spec.revision


def prepare_models(settings: PaddleSettings) -> dict[str, Any]:
    """Fetch only immutable model snapshots, never a floating automatic fallback."""
    from huggingface_hub import snapshot_download

    for role, spec in settings.models().items():
        snapshot_download(
            repo_id=spec.repository,
            revision=spec.revision,
            local_dir=model_directory(settings, role),
            allow_patterns=["inference.*", "config.json"],
        )
    return model_identity(settings)


def model_identity(settings: PaddleSettings) -> dict[str, Any]:
    identities = {}
    for role, spec in settings.models().items():
        directory = model_directory(settings, role)
        required = ("inference.json", "inference.yml", "inference.pdiparams")
        for name in required:
            if not (directory / name).is_file():
                raise ValueError(f"missing model {directory / name}; run prepare-models first")
        hashes = {p.name: sha256_file(p) for p in sorted(directory.iterdir()) if p.is_file()}
        identities[role] = {**spec.model_dump(), "files": hashes}
    return identities


def runtime_identity() -> dict[str, str]:
    versions = {}
    for name in ("paddleocr", "paddlex", "numpy", "pypdfium2", "Pillow", "pyarrow"):
        versions[name] = importlib.metadata.version(name)
    for name in ("paddlepaddle", "paddlepaddle-gpu"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    engines = set(versions) & {"paddlepaddle", "paddlepaddle-gpu"}
    if len(engines) != 1:
        raise ValueError("install exactly one Paddle runtime: CPU or GPU")
    expected = {k: v for k, v in RUNTIME_VERSIONS.items() if k != "paddle"}
    expected[next(iter(engines))] = RUNTIME_VERSIONS["paddle"]
    for name, version in expected.items():
        if versions[name] != version:
            raise ValueError(f"unsupported {name}: {versions[name]}; expected {version}")
    return versions


def _polygon(value: Any, width: int, height: int) -> list[list[float]]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError("a text polygon needs at least three vertices")
    result = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("invalid polygon vertex")
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError("non-numeric polygon coordinate")
        if not 0 <= x <= width or not 0 <= y <= height:
            raise ValueError("polygon outside its declared coordinate image")
        result.append([float(x), float(y)])
    return result


def normalize_result(native: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Pair recognition-filtered arrays; never zip text against unfiltered detections."""
    if "error" in native:
        raise ValueError(f"PaddleOCR error: {native['error']}")
    required = ("dt_polys", "rec_texts", "rec_scores", "rec_polys", "rec_boxes")
    if any(not isinstance(native.get(key), list) for key in required):
        raise ValueError("PaddleOCR result is missing its required arrays")
    count = len(native["rec_texts"])
    if any(len(native[key]) != count for key in required[2:]):
        raise ValueError("recognition texts, scores and geometry have different lengths")
    detections = [_polygon(p, width, height) for p in native["dt_polys"]]
    indices: dict[str, deque[int]] = defaultdict(deque)
    for index, poly in enumerate(detections):
        indices[canonical_json_sha256(poly)].append(index)
    lines = []
    for index in range(count):
        text = native["rec_texts"][index]
        score = native["rec_scores"][index]
        if not isinstance(text, str) or not isinstance(score, (float, int)) or not 0 <= score <= 1:
            raise ValueError("invalid recognition text or confidence")
        poly = _polygon(native["rec_polys"][index], width, height)
        matches = indices[canonical_json_sha256(poly)]
        if not matches:
            raise ValueError("recognized polygon has no corresponding detected region")
        box = native["rec_boxes"][index]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("invalid recognition box")
        x1, y1, x2, y2 = box
        if not (0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height):
            raise ValueError("recognition box outside coordinate image")
        lines.append(
            {
                "index": index,
                "detection_index": matches.popleft(),
                "text": text,
                "recognition_score": float(score),
                "polygon": poly,
                "bbox": box,
            }
        )
    return {
        "coordinate_system": "pixel_xy_top_left",
        "width": width,
        "height": height,
        "granularity": "text_line",
        "detections": detections,
        "lines": lines,
        "unrecognized_detection_indices": sorted(i for queue in indices.values() for i in queue),
    }


class PaddleBackend:
    """One persistent model instance, called serially while render workers prefetch."""

    def __init__(self, settings: PaddleSettings) -> None:
        self.settings = settings
        self.versions = runtime_identity()
        self.models = model_identity(settings)
        # All model directories are explicit. No remote model-source probing is needed.
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        paddle = import_module("paddle")
        PaddleOCR = import_module("paddleocr").PaddleOCR

        if settings.device.startswith("gpu:"):
            index = int(settings.device.split(":")[1])
            if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() <= index:
                raise ValueError(f"requested {settings.device} is unavailable; no CPU fallback")
        options: dict[str, Any] = {
            "device": settings.device,
            "engine": "paddle_static",
            "enable_hpi": False,
            "use_tensorrt": False,
            "enable_cinn": False,
            "cpu_threads": settings.cpu_threads,
            "enable_mkldnn": settings.enable_mkldnn,
            "mkldnn_cache_capacity": settings.mkldnn_cache_capacity,
            "text_recognition_batch_size": settings.recognition_batch_size,
            "textline_orientation_batch_size": settings.orientation_batch_size,
            "use_doc_orientation_classify": settings.document_orientation is not None,
            "use_doc_unwarping": settings.document_unwarping is not None,
            "use_textline_orientation": settings.textline_orientation is not None,
            "text_det_limit_side_len": settings.detection_limit_side_len,
            "text_det_limit_type": settings.detection_limit_type,
            "text_det_thresh": settings.detection_threshold,
            "text_det_box_thresh": settings.detection_box_threshold,
            "text_det_unclip_ratio": settings.detection_unclip_ratio,
            "text_rec_score_thresh": settings.recognition_score_threshold,
            "return_word_box": settings.return_word_boxes,
            "paddlex_config": {
                "pipeline_name": "OCR",
                "text_type": "general",
                "batch_size": 1,
                "SubModules": {
                    "TextDetection": {
                        "module_name": "text_detection",
                        "max_side_limit": settings.detection_max_side_limit,
                    },
                    "TextRecognition": {"module_name": "text_recognition"},
                    "TextLineOrientation": {"module_name": "textline_orientation"},
                },
                "SubPipelines": {
                    "DocPreprocessor": {
                        "pipeline_name": "doc_preprocessor",
                        "SubModules": {
                            "DocOrientationClassify": {"module_name": "doc_text_orientation"},
                            "DocUnwarping": {"module_name": "image_unwarping"},
                        },
                    }
                },
            },
        }
        prefixes = {
            "detection": "text_detection",
            "recognition": "text_recognition",
            "document_orientation": "doc_orientation_classify",
            "document_unwarping": "doc_unwarping",
            "textline_orientation": "textline_orientation",
        }
        for role, spec in settings.models().items():
            options[f"{prefixes[role]}_model_name"] = spec.name
            options[f"{prefixes[role]}_model_dir"] = str(model_directory(settings, role))
        self.engine = PaddleOCR(**options)

    def close(self) -> None:
        self.engine.close()

    def recognize(self, raster: Path, artifact_root: Path) -> dict[str, Any]:
        """Save native data and the actual coordinate image, not OCR-rendered text."""
        iterator = iter(self.engine.predict_iter(str(raster)))
        result = next(iterator, None)
        if result is None or next(iterator, None) is not None:
            raise ValueError("one raster must produce exactly one OCR page")
        if "error" in result:
            raise ValueError(f"PaddleOCR failed: {result['error']}")
        native = result.json["res"]
        if self.settings.return_word_boxes:
            for key in ("text_word", "text_word_region"):
                if key not in result:
                    raise ValueError(f"word geometry requested but {key} absent")
                # Paddle's JSON exporter omits word regions in this pinned release.
                native[key] = _json_value(result[key])
        coordinate_array = result["doc_preprocessor_res"]["output_img"]
        height, width = coordinate_array.shape[:2]
        normalized = normalize_result(native, width, height)
        transformed = (
            self.settings.document_orientation is not None
            or self.settings.document_unwarping is not None
        )
        coordinate_image = raster
        if transformed:
            coordinate_image = artifact_root / "coordinate-image.png"
            with Image.fromarray(coordinate_array[:, :, ::-1]) as img:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
            atomic_publish_bytes(coordinate_image, buffer.getvalue())
        native_path = artifact_root / "native.json"
        atomic_publish_json(native_path, native)
        return {
            **normalized,
            "coordinate_frame": "preprocessed_page" if transformed else "rendered_page",
            "coordinate_image_path": str(coordinate_image),
            "coordinate_image_sha256": sha256_file(coordinate_image),
            "native_path": str(native_path),
            "native_sha256": sha256_file(native_path),
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    # Reject unsupported native values rather than stringifying them.
    json.dumps(value, allow_nan=False)
    return value
