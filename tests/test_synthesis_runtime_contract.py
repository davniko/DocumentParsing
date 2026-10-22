"""The copied-code synthesis image must include its positional-stage dependency."""

import hashlib
import json
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def test_positions_arrow_dependency_is_pinned_in_the_synthesis_runtime():
    runtime = tomllib.loads((ROOT / "environments/synthesis/pyproject.toml").read_text())
    requirements = {
        value.name: value
        for item in runtime["project"]["dependencies"]
        if (value := Requirement(item))
    }
    lock = tomllib.loads((ROOT / "environments/synthesis/uv.lock").read_text())
    arrow = next(item for item in lock["package"] if item["name"] == "pyarrow")
    assert str(requirements["pyarrow"].specifier) == f"=={arrow['version']}"
    assert arrow["wheels"]


def test_synthesis_image_smoke_checks_the_real_positions_entrypoint():
    dockerfile = (ROOT / "docker/synthesis/Dockerfile").read_text()
    assert "from document_ocr.synthesis.curated_campaign import create_campaign" in dockerfile
    assert "from document_ocr.synthesis.curated_positions import position_campaign" in dockerfile


def test_pinned_reflow_font_is_available_to_the_image_build():
    directory = ROOT / "docker/synthesis/fonts"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert hashlib.sha256((directory / manifest["filename"]).read_bytes()).hexdigest() == manifest[
        "sha256"
    ]
    assert (directory / manifest["license"]).is_file()
    dockerfile = (ROOT / "docker/synthesis/Dockerfile").read_text()
    assert "COPY docker/synthesis/fonts/ /usr/share/fonts/truetype/dejavu/" in dockerfile
    assert "FontMetrics(font," in dockerfile
