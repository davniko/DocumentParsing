"""Process-isolated, bounded PDFium rasterization for OCR page inputs.

PDFium is not thread-safe.  The public functions in this module therefore bind
their process-local document cache to the first calling thread and reject calls
from any other thread in that process.  Parallel callers must use processes
(for example, a ``ProcessPoolExecutor``), never a thread pool.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import pypdfium2 as pdfium
from PIL import Image

from document_ocr.config import RasterConfig
from document_ocr.models import RasterMetadata

type PdfFormType = Literal[
    "none",
    "acroform",
    "xfa_full",
    "xfa_foreground",
]
type PdfBox = tuple[float, float, float, float]

_HASH_CHUNK_SIZE = 1024 * 1024
_VALID_ROTATIONS = {0, 90, 180, 270}


class _BinaryReader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


class PdfRendererError(RuntimeError):
    """Base class for explicit PDF inspection and rasterization failures."""


class RendererThreadSafetyError(PdfRendererError):
    """Raised when two threads attempt to share a process-local PDFium cache."""


class PdfOpenError(PdfRendererError):
    """Raised when a source cannot be opened as a PDF."""


class PdfLimitError(PdfRendererError):
    """Raised when a source exceeds a configured byte, page, or pixel bound."""


class PdfPageIndexError(PdfRendererError):
    """Raised when a zero-based page index is outside the source document."""


class UnsupportedPdfError(PdfRendererError):
    """Raised when the PDF needs a feature that is deliberately unsupported."""


class SourceChangedError(PdfRendererError):
    """Raised when the source identity changes while a page is being rendered."""


class PdfRenderError(PdfRendererError):
    """Raised when PDFium rendering or deterministic image encoding fails."""


@dataclass(frozen=True, slots=True)
class PdfInspection:
    """Bounded document-level facts obtained before scheduling page work."""

    canonical_path: str
    source_device: int
    source_inode: int
    source_size_bytes: int
    source_mtime_ns: int
    source_ctime_ns: int
    page_count: int
    pdf_version: int | None
    form_type: PdfFormType


@dataclass(frozen=True, slots=True)
class PageInspection:
    """Page geometry and source features used by one raster operation.

    Width and height are PDFium's displayed page dimensions, so intrinsic page
    rotation is already reflected in them.  Boxes are reported separately in
    source PDF canvas coordinates for diagnostics and audit probes.
    """

    page_index: int
    page_number: int
    width_points: float
    height_points: float
    rotation_degrees: Literal[0, 90, 180, 270]
    media_box: PdfBox | None
    crop_box: PdfBox | None
    bleed_box: PdfBox | None
    trim_box: PdfBox | None
    art_box: PdfBox | None
    annotation_count: int


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """Pickle-safe result returned by a renderer worker process."""

    output_path: str
    document: PdfInspection
    page: PageInspection
    raster: RasterMetadata


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    canonical_path: str
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int


@dataclass(slots=True)
class _CachedDocument:
    identity: _FileIdentity
    document: pdfium.PdfDocument
    page_count: int
    pdf_version: int | None
    form_type: PdfFormType

    def close(self) -> None:
        self.document.close()


_owner_pid: int | None = None
_owner_thread_id: int | None = None
_document_cache: OrderedDict[str, _CachedDocument] = OrderedDict()


def _ensure_process_thread_owner() -> None:
    """Bind PDFium state to one thread, resetting inherited state after fork."""

    global _document_cache, _owner_pid, _owner_thread_id

    pid = os.getpid()
    thread_id = threading.get_ident()
    if _owner_pid is None:
        _owner_pid = pid
        _owner_thread_id = thread_id
        return
    if _owner_pid != pid:
        # Forked PDFium handles must neither be reused nor closed in the child.
        # Dropping the copied Python references leaves the parent-owned handles
        # untouched; the child starts with a fresh cache.
        _document_cache = OrderedDict()
        _owner_pid = pid
        _owner_thread_id = thread_id
        return
    if _owner_thread_id != thread_id:
        raise RendererThreadSafetyError(
            "pypdfium2/PDFium state is process-local and bound to one thread; "
            "use a process pool, not a thread pool, for parallel rendering"
        )


def close_process_document_cache() -> None:
    """Close every PDF handle owned by the current renderer process."""

    _ensure_process_thread_owner()
    while _document_cache:
        _, entry = _document_cache.popitem(last=False)
        entry.close()


def _file_identity(pdf_path: str | Path) -> _FileIdentity:
    path = Path(pdf_path)
    if not path.is_absolute():
        raise PdfOpenError("PDF path must be absolute for reproducible worker execution")
    absolute = Path(os.path.abspath(path))
    try:
        canonical_path = path.resolve(strict=True)
        stat = canonical_path.stat()
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise PdfOpenError(f"cannot stat PDF source {path}: {exc}") from exc
    if canonical_path != absolute:
        raise PdfOpenError(f"PDF source must not traverse symbolic links: {path}")
    if not canonical_path.is_file():
        raise PdfOpenError(f"PDF source is not a regular file: {canonical_path}")
    if stat.st_size <= 0:
        raise PdfOpenError(f"PDF source is empty: {canonical_path}")
    return _FileIdentity(
        canonical_path=str(canonical_path),
        device=stat.st_dev,
        inode=stat.st_ino,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
    )


def _validate_file_limit(identity: _FileIdentity, config: RasterConfig) -> None:
    if identity.size_bytes > config.max_pdf_bytes:
        raise PdfLimitError(
            f"PDF is {identity.size_bytes} bytes, exceeding max_pdf_bytes="
            f"{config.max_pdf_bytes}: {identity.canonical_path}"
        )


def _map_form_type(raw_form_type: int) -> PdfFormType:
    form_types: dict[int, PdfFormType] = {
        pdfium.raw.FORMTYPE_NONE: "none",
        pdfium.raw.FORMTYPE_ACRO_FORM: "acroform",
        pdfium.raw.FORMTYPE_XFA_FULL: "xfa_full",
        pdfium.raw.FORMTYPE_XFA_FOREGROUND: "xfa_foreground",
    }
    try:
        return form_types[raw_form_type]
    except KeyError as exc:
        raise UnsupportedPdfError(f"PDFium returned unknown form type {raw_form_type}") from exc


def _validate_form_policy(
    form_type: PdfFormType,
    reject_xfa: bool,
) -> None:
    is_xfa = form_type in {"xfa_full", "xfa_foreground"}
    if is_xfa and reject_xfa:
        raise UnsupportedPdfError(
            f"document form type is {form_type}, and raster.reject_xfa is enabled"
        )
    if is_xfa and "XFA" not in pdfium.PDFIUM_INFO.flags:
        raise UnsupportedPdfError(
            f"document form type is {form_type}, but the installed PDFium binary "
            "does not include XFA support"
        )


def _initialize_forms(
    document: pdfium.PdfDocument,
    form_type: PdfFormType,
    reject_xfa: bool,
) -> None:
    _validate_form_policy(form_type, reject_xfa)
    if form_type != "none":
        document.init_forms()


def _evict_to_capacity(capacity: int) -> None:
    while len(_document_cache) > capacity:
        _, entry = _document_cache.popitem(last=False)
        entry.close()


def _open_cached_document(pdf_path: str | Path, config: RasterConfig) -> _CachedDocument:
    _ensure_process_thread_owner()
    identity = _file_identity(pdf_path)
    _validate_file_limit(identity, config)

    cached = _document_cache.get(identity.canonical_path)
    if cached is not None and cached.identity == identity:
        _validate_form_policy(cached.form_type, config.reject_xfa)
        _document_cache.move_to_end(identity.canonical_path)
        _validate_document_limits(cached, config)
        _evict_to_capacity(config.max_cached_documents_per_process)
        return cached
    if cached is not None:
        del _document_cache[identity.canonical_path]
        cached.close()

    document: pdfium.PdfDocument | None = None
    try:
        document = pdfium.PdfDocument(identity.canonical_path)
        form_type = _map_form_type(document.get_formtype())
        _initialize_forms(document, form_type, config.reject_xfa)
        page_count = len(document)
        pdf_version = document.get_version()
    except PdfRendererError:
        if document is not None:
            document.close()
        raise
    except Exception as exc:
        if document is not None:
            document.close()
        raise PdfOpenError(
            f"PDFium could not open or inspect {identity.canonical_path}: {exc}"
        ) from exc

    entry = _CachedDocument(
        identity=identity,
        document=document,
        page_count=page_count,
        pdf_version=pdf_version,
        form_type=form_type,
    )
    try:
        _validate_document_limits(entry, config)
    except BaseException:
        entry.close()
        raise
    _document_cache[identity.canonical_path] = entry
    _evict_to_capacity(config.max_cached_documents_per_process)
    return entry


def _validate_document_limits(entry: _CachedDocument, config: RasterConfig) -> None:
    if entry.page_count <= 0:
        raise PdfOpenError(f"PDF has no pages: {entry.identity.canonical_path}")
    if entry.page_count > config.max_pages_per_document:
        raise PdfLimitError(
            f"PDF has {entry.page_count} pages, exceeding max_pages_per_document="
            f"{config.max_pages_per_document}: {entry.identity.canonical_path}"
        )


def _document_inspection(entry: _CachedDocument) -> PdfInspection:
    return PdfInspection(
        canonical_path=entry.identity.canonical_path,
        source_device=entry.identity.device,
        source_inode=entry.identity.inode,
        source_size_bytes=entry.identity.size_bytes,
        source_mtime_ns=entry.identity.mtime_ns,
        source_ctime_ns=entry.identity.ctime_ns,
        page_count=entry.page_count,
        pdf_version=entry.pdf_version,
        form_type=entry.form_type,
    )


def inspect_pdf(pdf_path: str | Path, config: RasterConfig) -> PdfInspection:
    """Open a bounded PDF in the process cache and return document-level facts."""

    return _document_inspection(_open_cached_document(pdf_path, config))


def _validate_page_index(page_index: int, page_count: int) -> None:
    if isinstance(page_index, bool) or not isinstance(page_index, int):
        raise TypeError("page_index must be an integer")
    if not 0 <= page_index < page_count:
        raise PdfPageIndexError(
            f"page_index={page_index} is outside zero-based page range [0, {page_count})"
        )


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise PdfRenderError(f"PDFium returned non-finite {name}: {result}")
    return result


def _page_box(page: pdfium.PdfPage, getter_name: str) -> PdfBox | None:
    raw_box = getattr(page, getter_name)(fallback_ok=False)
    if raw_box is None:
        return None
    box = tuple(_finite_float(value, getter_name) for value in raw_box)
    if len(box) != 4:
        raise PdfRenderError(f"PDFium returned malformed {getter_name}: {raw_box!r}")
    return box


def _inspect_open_page(page: pdfium.PdfPage, page_index: int) -> PageInspection:
    width_points, height_points = (
        _finite_float(value, "displayed page size") for value in page.get_size()
    )
    if width_points <= 0 or height_points <= 0:
        raise PdfRenderError(
            f"page {page_index} has non-positive displayed dimensions "
            f"{width_points} x {height_points} points"
        )
    rotation = page.get_rotation()
    if rotation not in _VALID_ROTATIONS:
        raise PdfRenderError(
            f"PDFium returned unsupported page rotation {rotation} for page {page_index}"
        )
    annotation_count = int(pdfium.raw.FPDFPage_GetAnnotCount(page))
    if annotation_count < 0:
        raise PdfRenderError(f"PDFium failed to count annotations for page {page_index}")
    return PageInspection(
        page_index=page_index,
        page_number=page_index + 1,
        width_points=width_points,
        height_points=height_points,
        rotation_degrees=cast(Literal[0, 90, 180, 270], rotation),
        media_box=_page_box(page, "get_mediabox"),
        crop_box=_page_box(page, "get_cropbox"),
        bleed_box=_page_box(page, "get_bleedbox"),
        trim_box=_page_box(page, "get_trimbox"),
        art_box=_page_box(page, "get_artbox"),
        annotation_count=annotation_count,
    )


def inspect_page(
    pdf_path: str | Path,
    page_index: int,
    config: RasterConfig,
) -> tuple[PdfInspection, PageInspection]:
    """Return document and page facts without rasterizing the page."""

    entry = _open_cached_document(pdf_path, config)
    _validate_page_index(page_index, entry.page_count)
    page: pdfium.PdfPage | None = None
    try:
        page = entry.document[page_index]
        page_inspection = _inspect_open_page(page, page_index)
    except PdfRendererError:
        raise
    except Exception as exc:
        raise PdfRenderError(
            f"PDFium could not inspect page {page_index} of {entry.identity.canonical_path}: {exc}"
        ) from exc
    finally:
        if page is not None:
            page.close()
    return _document_inspection(entry), page_inspection


def _raster_dimensions(width_points: float, height_points: float, scale: float) -> tuple[int, int]:
    return math.ceil(width_points * scale), math.ceil(height_points * scale)


def _scale_is_bounded(
    width_points: float,
    height_points: float,
    scale: float,
    config: RasterConfig,
) -> bool:
    width_px, height_px = _raster_dimensions(width_points, height_points, scale)
    return (
        width_px >= 1
        and height_px >= 1
        and max(width_px, height_px) <= config.max_side_pixels
        and width_px * height_px <= config.max_pixels
    )


def _select_render_scale(
    width_points: float,
    height_points: float,
    config: RasterConfig,
) -> float:
    requested_scale = config.dpi / 72.0
    if _scale_is_bounded(width_points, height_points, requested_scale, config):
        return requested_scale

    high = min(
        requested_scale,
        config.max_side_pixels / max(width_points, height_points),
        math.sqrt(config.max_pixels / (width_points * height_points)),
    )
    if _scale_is_bounded(width_points, height_points, high, config):
        return high

    # Pixel dimensions use ceil(), so the continuous bounds above can exceed a
    # limit by one row or column.  A fixed-iteration binary search selects the
    # highest representable bounded scale deterministically.
    low = 0.0
    for _ in range(64):
        midpoint = (low + high) / 2.0
        if _scale_is_bounded(width_points, height_points, midpoint, config):
            low = midpoint
        else:
            high = midpoint
    if low <= 0.0:
        raise PdfLimitError("configured raster bounds cannot produce even a one-pixel page image")
    return low


def _expected_output_suffixes(image_format: Literal["png", "jpeg"]) -> set[str]:
    return {".png"} if image_format == "png" else {".jpg", ".jpeg"}


def _validate_output_path(
    output_path: str | Path,
    source_path: str,
    image_format: Literal["png", "jpeg"],
) -> Path:
    path = Path(output_path)
    if not path.is_absolute():
        raise PdfRenderError("raster output path must be absolute")
    absolute = Path(os.path.abspath(path))
    resolved = path.resolve(strict=False)
    if resolved != absolute:
        raise PdfRenderError(f"raster output path must not traverse symbolic links: {path}")
    if str(resolved) == source_path:
        raise PdfRenderError("raster output path must not overwrite the source PDF")
    if resolved.suffix.lower() not in _expected_output_suffixes(image_format):
        expected = ", ".join(sorted(_expected_output_suffixes(image_format)))
        raise PdfRenderError(
            f"raster output suffix must match image_format={image_format}; "
            f"expected one of {expected}"
        )
    return resolved


def _sha256_stream(stream: _BinaryReader) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(_HASH_CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def _write_image_atomic(
    image: Image.Image,
    output_path: Path,
    config: RasterConfig,
    before_publish: Callable[[], None],
) -> tuple[int, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            if config.image_format == "png":
                image.save(
                    stream,
                    format="PNG",
                    optimize=False,
                    compress_level=6,
                )
            else:
                image.save(
                    stream,
                    format="JPEG",
                    quality=config.jpeg_quality,
                    subsampling=0,
                    optimize=False,
                    progressive=False,
                )
            stream.flush()
            os.fsync(stream.fileno())
            size_bytes = stream.tell()
            if size_bytes <= 0:
                raise PdfRenderError("image encoder produced an empty raster")
            stream.seek(0)
            raster_sha256 = _sha256_stream(stream)

        before_publish()
        os.replace(temporary_path, output_path)
        temporary_path = None
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return size_bytes, raster_sha256
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_source_unchanged(identity: _FileIdentity) -> None:
    try:
        current = _file_identity(identity.canonical_path)
    except PdfOpenError as exc:
        raise SourceChangedError(
            f"PDF source became unavailable during rasterization: {identity.canonical_path}"
        ) from exc
    if current != identity:
        stale = _document_cache.pop(identity.canonical_path, None)
        if stale is not None:
            stale.close()
        raise SourceChangedError(
            f"PDF source changed during rasterization: {identity.canonical_path}"
        )


def render_page(
    pdf_path: str | Path,
    page_index: int,
    output_path: str | Path,
    config: RasterConfig,
) -> RenderedPage:
    """Rasterize one page deterministically and atomically publish its image.

    This synchronous function is deliberately suitable for direct submission
    to a process pool.  The bounded process-local cache amortizes PDF parsing
    when the worker receives more pages from the same document.
    """

    entry = _open_cached_document(pdf_path, config)
    _validate_page_index(page_index, entry.page_count)
    resolved_output_path = _validate_output_path(
        output_path,
        entry.identity.canonical_path,
        config.image_format,
    )

    page: pdfium.PdfPage | None = None
    bitmap: pdfium.PdfBitmap | None = None
    pil_image: Image.Image | None = None
    rgb_image: Image.Image | None = None
    started = time.perf_counter()
    try:
        page = entry.document[page_index]
        page_inspection = _inspect_open_page(page, page_index)
        scale = _select_render_scale(
            page_inspection.width_points,
            page_inspection.height_points,
            config,
        )
        expected_width, expected_height = _raster_dimensions(
            page_inspection.width_points,
            page_inspection.height_points,
            scale,
        )
        bitmap = page.render(
            scale=scale,
            rotation=0,
            may_draw_forms=True,
            fill_color=(255, 255, 255, 255),
            draw_annots=config.draw_annotations,
            limit_image_cache=True,
            rev_byteorder=True,
        )
        if (bitmap.width, bitmap.height) != (expected_width, expected_height):
            raise PdfRenderError(
                "PDFium raster dimensions differed from the validated bounds: "
                f"expected {expected_width}x{expected_height}, got "
                f"{bitmap.width}x{bitmap.height}"
            )
        if bitmap.width * bitmap.height > config.max_pixels:
            raise PdfLimitError(
                f"rendered page contains {bitmap.width * bitmap.height} pixels, "
                f"exceeding max_pixels={config.max_pixels}"
            )

        pil_image = bitmap.to_pil()
        rgb_image = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        size_bytes, raster_sha256 = _write_image_atomic(
            rgb_image,
            resolved_output_path,
            config,
            lambda: _validate_source_unchanged(entry.identity),
        )
        duration_ms = (time.perf_counter() - started) * 1000.0
    except PdfRendererError:
        raise
    except Exception as exc:
        raise PdfRenderError(
            f"failed to rasterize page {page_index} of {entry.identity.canonical_path}: {exc}"
        ) from exc
    finally:
        if rgb_image is not None and rgb_image is not pil_image:
            rgb_image.close()
        if pil_image is not None:
            pil_image.close()
        if bitmap is not None:
            bitmap.close()
        if page is not None:
            page.close()

    mime_type: Literal["image/png", "image/jpeg"] = (
        "image/png" if config.image_format == "png" else "image/jpeg"
    )
    raster_metadata = RasterMetadata(
        renderer_name="pypdfium2",
        renderer_version=str(pdfium.PYPDFIUM_INFO),
        pdfium_version=str(pdfium.PDFIUM_INFO),
        page_width_points=page_inspection.width_points,
        page_height_points=page_inspection.height_points,
        page_rotation_degrees=page_inspection.rotation_degrees,
        requested_dpi=config.dpi,
        effective_dpi=scale * 72.0,
        render_scale=scale,
        raster_width_px=expected_width,
        raster_height_px=expected_height,
        raster_image_format=config.image_format,
        raster_mime_type=mime_type,
        draw_annotations=config.draw_annotations,
        pdf_form_type=entry.form_type,
        raster_size_bytes=size_bytes,
        raster_sha256=raster_sha256,
        render_duration_ms=duration_ms,
    )
    return RenderedPage(
        output_path=str(resolved_output_path),
        document=_document_inspection(entry),
        page=page_inspection,
        raster=raster_metadata,
    )
