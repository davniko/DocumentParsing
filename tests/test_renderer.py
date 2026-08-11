from __future__ import annotations

import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.annotations import FreeText
from pypdf.generic import DictionaryObject, NameObject, TextStringObject

import document_ocr.renderer as renderer
from document_ocr.config import RasterConfig
from document_ocr.renderer import (
    PdfLimitError,
    PdfOpenError,
    PdfPageIndexError,
    PdfRenderError,
    RendererThreadSafetyError,
    SourceChangedError,
    UnsupportedPdfError,
    close_process_document_cache,
    inspect_page,
    inspect_pdf,
    render_page,
)


def raster_config(
    *,
    dpi: int = 200,
    max_side_pixels: int = 4096,
    max_pixels: int = 12_000_000,
    max_pages_per_document: int = 500,
    max_pdf_bytes: int = 1_000_000_000,
    max_cached_documents_per_process: int = 2,
    image_format: Literal["png", "jpeg"] = "png",
    draw_annotations: bool = True,
    reject_xfa: bool = True,
) -> RasterConfig:
    return RasterConfig(
        dpi=dpi,
        max_side_pixels=max_side_pixels,
        max_pixels=max_pixels,
        max_pages_per_document=max_pages_per_document,
        max_pdf_bytes=max_pdf_bytes,
        max_cached_documents_per_process=max_cached_documents_per_process,
        image_format=image_format,
        jpeg_quality=95 if image_format == "jpeg" else None,
        draw_annotations=draw_annotations,
        reject_xfa=reject_xfa,
    )


def write_pdf(
    path: Path,
    page_sizes: tuple[tuple[float, float], ...] = ((200.0, 100.0),),
    *,
    first_page_rotation: int = 0,
    annotation: bool = False,
    form_type: Literal["none", "acroform", "xfa_foreground"] = "none",
) -> Path:
    writer = PdfWriter()
    for width, height in page_sizes:
        writer.add_blank_page(width=width, height=height)
    if first_page_rotation:
        writer.pages[0].rotate(first_page_rotation)
    if annotation:
        writer.add_annotation(
            page_number=0,
            annotation=FreeText(
                text="annotation text",
                rect=(10, 10, 90, 40),
                font_size="12pt",
            ),
        )
    if form_type != "none":
        acroform = DictionaryObject()
        if form_type == "xfa_foreground":
            acroform[NameObject("/XFA")] = TextStringObject("xfa-payload")
        writer._root_object[NameObject("/AcroForm")] = acroform
    with path.open("wb") as stream:
        writer.write(stream)
    return path


def _render_in_spawned_process(
    pdf_path: str,
    output_path: str,
    config: RasterConfig,
) -> tuple[int, int, str]:
    rendered = render_page(pdf_path, 0, output_path, config)
    return (
        rendered.raster.raster_width_px,
        rendered.raster.raster_height_px,
        rendered.raster.raster_sha256,
    )


@pytest.fixture(autouse=True)
def empty_process_cache() -> None:
    close_process_document_cache()
    yield
    close_process_document_cache()


def test_inspection_preserves_rotation_geometry_boxes_and_annotations(
    tmp_path: Path,
) -> None:
    pdf_path = write_pdf(
        tmp_path / "rotated.pdf",
        first_page_rotation=90,
        annotation=True,
    )

    document, page = inspect_page(pdf_path, 0, raster_config())

    assert document.page_count == 1
    source_stat = pdf_path.stat()
    assert document.source_device == source_stat.st_dev
    assert document.source_inode == source_stat.st_ino
    assert document.source_size_bytes == source_stat.st_size
    assert document.source_mtime_ns == source_stat.st_mtime_ns
    assert document.source_ctime_ns == source_stat.st_ctime_ns
    assert document.form_type == "none"
    assert document.canonical_path == str(pdf_path.resolve())
    assert page.page_index == 0
    assert page.page_number == 1
    assert page.rotation_degrees == 90
    assert page.width_points == pytest.approx(100.0)
    assert page.height_points == pytest.approx(200.0)
    assert page.media_box == pytest.approx((0.0, 0.0, 200.0, 100.0))
    assert page.crop_box is None
    assert page.annotation_count == 1


def test_png_render_is_rgb_bounded_and_byte_deterministic(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "large.pdf", ((612.0, 792.0),))
    config = raster_config(max_side_pixels=1_000, max_pixels=500_000)
    first_path = tmp_path / "page-first.png"
    second_path = tmp_path / "page-second.png"

    first = render_page(pdf_path, 0, first_path, config)
    second = render_page(pdf_path, 0, second_path, config)

    assert first.raster.raster_width_px * first.raster.raster_height_px <= 500_000
    assert max(first.raster.raster_width_px, first.raster.raster_height_px) <= 1_000
    assert first.raster.effective_dpi < config.dpi
    assert first.raster.raster_mime_type == "image/png"
    assert first.raster.draw_annotations is True
    assert first.raster.raster_sha256 == second.raster.raster_sha256
    assert first_path.read_bytes() == second_path.read_bytes()
    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == first.raster.raster_sha256
    assert first.raster.raster_size_bytes == first_path.stat().st_size
    with Image.open(first_path) as image:
        assert image.mode == "RGB"
        assert image.size == (
            first.raster.raster_width_px,
            first.raster.raster_height_px,
        )


def test_requested_dpi_is_retained_when_bounds_do_not_reduce_it(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "small.pdf")

    rendered = render_page(
        pdf_path,
        0,
        tmp_path / "small.png",
        raster_config(dpi=144),
    )

    assert rendered.raster.render_scale == pytest.approx(2.0)
    assert rendered.raster.effective_dpi == pytest.approx(144.0)
    assert (rendered.raster.raster_width_px, rendered.raster.raster_height_px) == (
        400,
        200,
    )


def test_jpeg_render_is_rgb_and_deterministic(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "page.pdf", annotation=True)
    config = raster_config(image_format="jpeg", draw_annotations=False)

    first = render_page(pdf_path, 0, tmp_path / "first.jpg", config)
    second = render_page(pdf_path, 0, tmp_path / "second.jpeg", config)

    assert first.raster.raster_mime_type == "image/jpeg"
    assert first.raster.draw_annotations is False
    assert first.raster.raster_sha256 == second.raster.raster_sha256
    with Image.open(first.output_path) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"


def test_byte_and_page_limits_fail_before_page_render(tmp_path: Path) -> None:
    pdf_path = write_pdf(
        tmp_path / "two-pages.pdf",
        ((100.0, 100.0), (100.0, 100.0)),
    )

    with pytest.raises(PdfLimitError, match="max_pdf_bytes"):
        inspect_pdf(pdf_path, raster_config(max_pdf_bytes=pdf_path.stat().st_size - 1))
    with pytest.raises(PdfLimitError, match="max_pages_per_document"):
        inspect_pdf(pdf_path, raster_config(max_pages_per_document=1))


@pytest.mark.parametrize("page_index", [-1, 1])
def test_page_index_must_be_inside_zero_based_range(
    tmp_path: Path,
    page_index: int,
) -> None:
    pdf_path = write_pdf(tmp_path / "one.pdf")

    with pytest.raises(PdfPageIndexError, match="outside zero-based page range"):
        inspect_page(pdf_path, page_index, raster_config())


@pytest.mark.parametrize("page_index", [True, 0.0])
def test_page_index_rejects_bool_and_non_integer(
    tmp_path: Path,
    page_index: object,
) -> None:
    pdf_path = write_pdf(tmp_path / "one.pdf")

    with pytest.raises(TypeError, match="must be an integer"):
        inspect_page(pdf_path, page_index, raster_config())  # type: ignore[arg-type]


def test_invalid_sources_and_output_contracts_surface_errors(tmp_path: Path) -> None:
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"not a PDF")
    valid_path = write_pdf(tmp_path / "valid.pdf")

    with pytest.raises(PdfOpenError, match="PDFium could not open"):
        inspect_pdf(corrupt_path, raster_config())
    with pytest.raises(PdfOpenError, match="must be absolute"):
        inspect_pdf(Path("relative.pdf"), raster_config())
    with pytest.raises(PdfRenderError, match="suffix must match"):
        render_page(valid_path, 0, tmp_path / "wrong.jpg", raster_config())
    with pytest.raises(PdfRenderError, match="must not overwrite"):
        render_page(valid_path, 0, valid_path, raster_config())


def test_source_and_output_paths_never_traverse_symlinks(tmp_path: Path) -> None:
    source = write_pdf(tmp_path / "source.pdf")
    source_alias = tmp_path / "source-alias.pdf"
    source_alias.symlink_to(source)
    with pytest.raises(PdfOpenError, match="must not traverse symbolic links"):
        inspect_pdf(source_alias, raster_config())

    output_directory = tmp_path / "actual-output"
    output_directory.mkdir()
    output_alias = tmp_path / "output-alias"
    output_alias.symlink_to(output_directory, target_is_directory=True)
    with pytest.raises(PdfRenderError, match="must not traverse symbolic links"):
        render_page(source, 0, output_alias / "page.png", raster_config())
    assert list(output_directory.iterdir()) == []


def test_xfa_is_rejected_and_unsupported_binary_never_silently_degrades(
    tmp_path: Path,
) -> None:
    pdf_path = write_pdf(tmp_path / "xfa.pdf", form_type="xfa_foreground")

    with pytest.raises(UnsupportedPdfError, match="reject_xfa is enabled"):
        inspect_pdf(pdf_path, raster_config(reject_xfa=True))
    with pytest.raises(UnsupportedPdfError, match="does not include XFA support"):
        inspect_pdf(pdf_path, raster_config(reject_xfa=False))


def test_cached_xfa_document_rechecks_stricter_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = write_pdf(tmp_path / "cached-xfa.pdf", form_type="xfa_foreground")
    monkeypatch.setattr(renderer, "_initialize_forms", lambda *_args: None)
    assert inspect_pdf(pdf_path, raster_config(reject_xfa=False)).form_type == "xfa_foreground"

    with pytest.raises(UnsupportedPdfError, match="reject_xfa is enabled"):
        inspect_pdf(pdf_path, raster_config(reject_xfa=True))


def test_acroform_type_is_recorded(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "acroform.pdf", form_type="acroform")

    inspection = inspect_pdf(pdf_path, raster_config())

    assert inspection.form_type == "acroform"


def test_process_local_cache_is_bounded_lru(tmp_path: Path) -> None:
    config = raster_config(max_cached_documents_per_process=2)
    paths = [write_pdf(tmp_path / f"{index}.pdf") for index in range(3)]

    for path in paths:
        inspect_pdf(path, config)

    assert len(renderer._document_cache) == 2
    assert list(renderer._document_cache) == [
        str(paths[1].resolve()),
        str(paths[2].resolve()),
    ]


def test_cache_invalidates_when_source_identity_changes(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "mutable.pdf")
    first = inspect_pdf(pdf_path, raster_config())
    with pdf_path.open("ab") as stream:
        stream.write(b"\n% identity change\n")

    second = inspect_pdf(pdf_path, raster_config())

    assert second.source_size_bytes > first.source_size_bytes
    assert len(renderer._document_cache) == 1


def test_atomic_publish_preserves_existing_output_if_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = write_pdf(tmp_path / "source.pdf")
    output_path = tmp_path / "page.png"
    output_path.write_bytes(b"existing-complete-artifact")

    def reject_publish(identity: object) -> None:
        raise SourceChangedError("simulated source mutation")

    monkeypatch.setattr(renderer, "_validate_source_unchanged", reject_publish)
    with pytest.raises(SourceChangedError, match="simulated source mutation"):
        render_page(pdf_path, 0, output_path, raster_config())

    assert output_path.read_bytes() == b"existing-complete-artifact"
    assert list(tmp_path.glob(".page.png.*.tmp")) == []


def test_thread_pool_sharing_is_explicitly_rejected(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "thread.pdf")
    inspect_pdf(pdf_path, raster_config())

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(inspect_pdf, pdf_path, raster_config())
        with pytest.raises(RendererThreadSafetyError, match="process pool"):
            future.result()


def test_render_function_runs_in_spawned_process(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "spawn.pdf")
    output_path = tmp_path / "spawn.png"

    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        width, height, raster_sha256 = executor.submit(
            _render_in_spawned_process,
            str(pdf_path),
            str(output_path),
            raster_config(dpi=72),
        ).result(timeout=30)

    assert (width, height) == (200, 100)
    assert hashlib.sha256(output_path.read_bytes()).hexdigest() == raster_sha256
