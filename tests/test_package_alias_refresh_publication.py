import importlib.util
from pathlib import Path

import pytest

from document_ocr.hashing import sha256_file


def publication_module():
    path = Path(__file__).resolve().parents[1] / "scripts/synthesis/refresh_r28_package_aliases.py"
    spec = importlib.util.spec_from_file_location("alias_refresh_publication", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    current, stage, before = (tmp_path / s for s in ("current", "stage", "before"))
    for directory in (current, stage, before):
        directory.mkdir()
    for name in ("first", "second"):
        (current / name).write_text("old " + name)
        (before / name).write_text("old " + name)
        (stage / name).write_text("new " + name)
    old = {name: sha256_file(current / name) for name in ("first", "second")}
    new = {name: sha256_file(stage / name) for name in ("first", "second")}
    return current, stage, before, old, new


def test_publication_and_precondition_check(tmp_path):
    module = publication_module()
    current, stage, before, old, new = fixture(tmp_path)
    module.promote_files(current, stage, before, old, new)
    assert {name: sha256_file(current / name) for name in new} == new
    with pytest.raises(ValueError, match="current file changed"):
        module.promote_files(current, stage, before, old, new)


def test_failed_promotion_restores_all_original_bytes(tmp_path, monkeypatch):
    module = publication_module()
    current, stage, before, old, new = fixture(tmp_path)
    replace = module.os.replace

    def fail_second(source, destination):
        if destination.name == "second":
            raise OSError("simulated interruption")
        return replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_second)
    with pytest.raises(OSError, match="simulated"):
        module.promote_files(current, stage, before, old, new)
    assert {name: sha256_file(current / name) for name in old} == old


def test_changed_stage_is_rejected_before_any_write(tmp_path):
    module = publication_module()
    current, stage, before, old, new = fixture(tmp_path)
    (stage / "second").write_text("unexpected")
    with pytest.raises(ValueError, match="staged file changed"):
        module.promote_files(current, stage, before, old, new)
    assert {name: sha256_file(current / name) for name in old} == old
