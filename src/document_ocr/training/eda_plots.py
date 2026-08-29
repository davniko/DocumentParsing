"""Deterministic Pillow charts for dataset EDA artifacts."""

from __future__ import annotations

import io
import math
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

_WIDTH = 1600
_HEIGHT = 1000
_BACKGROUND = "#F7F9FC"
_INK = "#172033"
_MUTED = "#62708A"
_GRID = "#DDE3ED"
_ACCENT = "#2563EB"
_PALETTE = (
    "#2563EB",
    "#06B6D4",
    "#10B981",
    "#F59E0B",
    "#EF4444",
    "#8B5CF6",
    "#EC4899",
    "#64748B",
)


@dataclass(frozen=True, slots=True)
class ScatterPoint:
    x: float
    y: float
    label: str = ""
    size: float = 1.0
    series: int = 0


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size=size)


def _canvas(title: str, subtitle: str, footnote: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (_WIDTH, _HEIGHT), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((70, 48), title, fill=_INK, font=_font(38, bold=True))
    if subtitle:
        draw.text((72, 101), subtitle, fill=_MUTED, font=_font(20))
    if footnote:
        draw.text((72, 957), footnote, fill=_MUTED, font=_font(15))
    return image, draw


def _png(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG", compress_level=6, optimize=False)
    return stream.getvalue()


def _short(value: str, width: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= width:
        return normalized
    return normalized[: max(1, width - 1)].rstrip() + "…"


def _number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def horizontal_bar(
    *,
    title: str,
    subtitle: str,
    items: Sequence[tuple[str, float]],
    total: float | None = None,
    footnote: str = "",
    color: str = _ACCENT,
) -> bytes:
    image, draw = _canvas(title, subtitle, footnote)
    left, right, top, bottom = 430, 1500, 155, 915
    values = [value for _, value in items]
    maximum = max(values, default=1.0) or 1.0
    row_height = (bottom - top) / max(1, len(items))
    label_font = _font(max(13, min(21, int(row_height * 0.36))))
    value_font = _font(max(12, min(19, int(row_height * 0.32))), bold=True)
    for index, (label, value) in enumerate(items):
        y0 = top + index * row_height + row_height * 0.17
        y1 = top + (index + 1) * row_height - row_height * 0.17
        draw.text(
            (left - 18, (y0 + y1) / 2),
            _short(label, 38),
            anchor="rm",
            fill=_INK,
            font=label_font,
        )
        draw.rounded_rectangle(
            (left, y0, left + (right - left) * value / maximum, y1),
            radius=6,
            fill=color,
        )
        percentage = f" · {value / total:.1%}" if total else ""
        draw.text(
            (left + (right - left) * value / maximum + 10, (y0 + y1) / 2),
            f"{_number(float(value))}{percentage}",
            anchor="lm",
            fill=_INK,
            font=value_font,
        )
    return _png(image)


def vertical_bar(
    *,
    title: str,
    subtitle: str,
    items: Sequence[tuple[str, float]],
    footnote: str = "",
    color: str = _ACCENT,
) -> bytes:
    image, draw = _canvas(title, subtitle, footnote)
    left, right, top, bottom = 115, 1510, 165, 850
    values = [value for _, value in items]
    maximum = max(values, default=1.0) or 1.0
    for grid_index in range(6):
        y = bottom - (bottom - top) * grid_index / 5
        value = maximum * grid_index / 5
        draw.line((left, y, right, y), fill=_GRID, width=1)
        draw.text((left - 12, y), _number(value), anchor="rm", fill=_MUTED, font=_font(15))
    slot = (right - left) / max(1, len(items))
    for index, (label, value) in enumerate(items):
        x0 = left + slot * index + slot * 0.14
        x1 = left + slot * (index + 1) - slot * 0.14
        y0 = bottom - (bottom - top) * value / maximum
        draw.rounded_rectangle((x0, y0, x1, bottom), radius=5, fill=color)
        draw.text(
            ((x0 + x1) / 2, y0 - 8),
            _number(float(value)),
            anchor="ms",
            fill=_INK,
            font=_font(15, bold=True),
        )
        wrapped = textwrap.wrap(_short(label, 22), width=max(5, min(14, int(slot / 10))))
        draw.multiline_text(
            ((x0 + x1) / 2, bottom + 14),
            "\n".join(wrapped[:3]),
            anchor="ma",
            align="center",
            fill=_INK,
            font=_font(max(11, min(16, int(slot / 8)))),
            spacing=3,
        )
    return _png(image)


def histogram(
    *,
    title: str,
    subtitle: str,
    values: Sequence[float],
    bins: int,
    footnote: str = "",
    color: str = _ACCENT,
) -> bytes:
    if not values:
        raise ValueError("histogram requires at least one value")
    low, high = min(values), max(values)
    if low == high:
        edges = [low - 0.5, high + 0.5]
    else:
        bin_count = max(1, min(bins, len(set(values))))
        step = (high - low) / bin_count
        edges = [low + step * index for index in range(bin_count + 1)]
    counts = [0] * (len(edges) - 1)
    for value in values:
        index = min(len(counts) - 1, int((value - edges[0]) / (edges[-1] - edges[0]) * len(counts)))
        counts[max(0, index)] += 1
    labels = [f"{edges[index]:.1f}-{edges[index + 1]:.1f}" for index in range(len(counts))]
    image, draw = _canvas(title, subtitle, footnote)
    left, right, top, bottom = 115, 1510, 165, 850
    maximum = max(counts) or 1
    for grid_index in range(6):
        y = bottom - (bottom - top) * grid_index / 5
        draw.line((left, y, right, y), fill=_GRID, width=1)
        draw.text(
            (left - 12, y),
            _number(float(maximum * grid_index / 5)),
            anchor="rm",
            fill=_MUTED,
            font=_font(15),
        )
    slot = (right - left) / len(counts)
    for index, count in enumerate(counts):
        x0 = left + slot * index + 1
        x1 = left + slot * (index + 1) - 1
        y0 = bottom - (bottom - top) * count / maximum
        draw.rectangle((x0, y0, x1, bottom), fill=color)
    tick_indexes = sorted(
        {0, len(labels) // 4, len(labels) // 2, 3 * len(labels) // 4, len(labels) - 1}
    )
    for index in tick_indexes:
        draw.text(
            (left + slot * (index + 0.5), bottom + 15),
            labels[index],
            anchor="ma",
            fill=_INK,
            font=_font(14),
        )
    return _png(image)


def line_chart(
    *,
    title: str,
    subtitle: str,
    series: Sequence[tuple[str, Sequence[tuple[float, float]]]],
    footnote: str = "",
    x_label: str = "",
    y_label: str = "",
) -> bytes:
    points = [point for _, values in series for point in values]
    if not points:
        raise ValueError("line chart requires points")
    image, draw = _canvas(title, subtitle, footnote)
    left, right, top, bottom = 130, 1490, 175, 850
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(0.0, min(ys)), max(ys)
    if x0 == x1:
        x1 = x0 + 1
    if y0 == y1:
        y1 = y0 + 1

    def px(value: float) -> float:
        return left + (right - left) * (value - x0) / (x1 - x0)

    def py(value: float) -> float:
        return bottom - (bottom - top) * (value - y0) / (y1 - y0)

    for index in range(6):
        y = top + (bottom - top) * index / 5
        value = y1 - (y1 - y0) * index / 5
        draw.line((left, y, right, y), fill=_GRID, width=1)
        draw.text((left - 12, y), _number(value), anchor="rm", fill=_MUTED, font=_font(15))
    for index, (name, values) in enumerate(series):
        color = _PALETTE[index % len(_PALETTE)]
        coords = [(px(x), py(y)) for x, y in values]
        if len(coords) > 1:
            draw.line(coords, fill=color, width=5, joint="curve")
        for x, y in coords:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
        draw.rounded_rectangle((1160, 155 + index * 34, 1190, 175 + index * 34), 5, fill=color)
        draw.text((1202, 165 + index * 34), name, anchor="lm", fill=_INK, font=_font(16))
    draw.text(((left + right) / 2, 910), x_label, anchor="mm", fill=_MUTED, font=_font(17))
    if y_label:
        draw.text((left, top - 18), y_label, anchor="ls", fill=_MUTED, font=_font(17))
    return _png(image)


def heatmap(
    *,
    title: str,
    subtitle: str,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    matrix: Sequence[Sequence[float]],
    footnote: str = "",
    value_format: str = "count",
    center_zero: bool = False,
) -> bytes:
    if not x_labels or not y_labels:
        raise ValueError("heatmap requires both axes")
    if len(matrix) != len(y_labels) or any(len(row) != len(x_labels) for row in matrix):
        raise ValueError("heatmap matrix shape differs from labels")
    image, draw = _canvas(title, subtitle, footnote)
    left, right, top, bottom = 300, 1510, 175, 830
    width = (right - left) / len(x_labels)
    height = (bottom - top) / len(y_labels)
    maximum = max((value for row in matrix for value in row), default=1.0) or 1.0
    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            if center_zero:
                bounded = max(-1.0, min(1.0, value))
                intensity = abs(bounded)
                base = (247, 249, 252)
                accent = (37, 99, 235) if bounded >= 0 else (220, 38, 38)
            else:
                intensity = math.sqrt(max(0.0, value) / maximum)
                base = (235, 241, 255)
                accent = (37, 99, 235)
            fill = tuple(round(a + (b - a) * intensity) for a, b in zip(base, accent, strict=True))
            x0 = left + x_index * width
            y0 = top + y_index * height
            draw.rectangle((x0, y0, x0 + width, y0 + height), fill=fill, outline=_BACKGROUND)
            if width >= 42 and height >= 28:
                if value_format == "percent":
                    label = f"{value:.0%}"
                elif value_format == "float":
                    label = f"{value:.2f}"
                else:
                    label = f"{int(value):,}"
                draw.text(
                    (x0 + width / 2, y0 + height / 2),
                    label,
                    anchor="mm",
                    fill="white" if intensity > 0.58 else _INK,
                    font=_font(max(11, min(16, int(min(width, height) * 0.34))), bold=True),
                )
    for index, label in enumerate(x_labels):
        draw.text(
            (left + (index + 0.5) * width, bottom + 13),
            _short(label, 16),
            anchor="ma",
            fill=_INK,
            font=_font(max(10, min(15, int(width / 7)))),
        )
    for index, label in enumerate(y_labels):
        draw.text(
            (left - 14, top + (index + 0.5) * height),
            _short(label, 28),
            anchor="rm",
            fill=_INK,
            font=_font(max(11, min(16, int(height * 0.36)))),
        )
    return _png(image)


def grouped_bar(
    *,
    title: str,
    subtitle: str,
    categories: Sequence[str],
    series: Sequence[tuple[str, Sequence[float]]],
    footnote: str = "",
    percent: bool = False,
) -> bytes:
    if not categories or not series:
        raise ValueError("grouped bar requires categories and series")
    if any(len(values) != len(categories) for _, values in series):
        raise ValueError("grouped bar series shape differs from categories")
    image, draw = _canvas(title, subtitle, footnote)
    left, right, top, bottom = 120, 1510, 190, 840
    maximum = max(value for _, values in series for value in values) or 1.0
    if percent:
        maximum = max(1.0, maximum)
    for index in range(6):
        y = bottom - (bottom - top) * index / 5
        value = maximum * index / 5
        draw.line((left, y, right, y), fill=_GRID, width=1)
        label = f"{value:.0%}" if percent else _number(value)
        draw.text((left - 10, y), label, anchor="rm", fill=_MUTED, font=_font(15))
    group_width = (right - left) / len(categories)
    bar_width = group_width * 0.72 / len(series)
    for category_index, category in enumerate(categories):
        center = left + group_width * (category_index + 0.5)
        start = center - bar_width * len(series) / 2
        for series_index, (_, values) in enumerate(series):
            value = values[category_index]
            x0 = start + series_index * bar_width
            y0 = bottom - (bottom - top) * value / maximum
            draw.rectangle((x0, y0, x0 + bar_width - 2, bottom), fill=_PALETTE[series_index])
        draw.text(
            (center, bottom + 14),
            _short(category, 18),
            anchor="ma",
            align="center",
            fill=_INK,
            font=_font(max(10, min(15, int(group_width / 8)))),
        )
    legend_width = min(230, 1040 // len(series))
    legend_start = right - legend_width * len(series)
    for index, (name, _) in enumerate(series):
        x = legend_start + index * legend_width
        draw.rounded_rectangle((x, 150, x + 28, 170), 4, fill=_PALETTE[index])
        draw.text((x + 38, 160), name, anchor="lm", fill=_INK, font=_font(15))
    return _png(image)


def scatter(
    *,
    title: str,
    subtitle: str,
    points: Sequence[ScatterPoint],
    footnote: str = "",
    x_label: str = "",
    y_label: str = "",
    label_top: int = 12,
) -> bytes:
    if not points:
        raise ValueError("scatter plot requires points")
    image, draw = _canvas(title, subtitle, footnote)
    left, right, top, bottom = 135, 1490, 175, 850
    xs, ys = [row.x for row in points], [row.y for row in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x0 == x1:
        x1 += 1
    if y0 == y1:
        y1 += 1
    x_pad = (x1 - x0) * 0.04
    y_pad = (y1 - y0) * 0.06
    x0, x1 = x0 - x_pad, x1 + x_pad
    y0, y1 = y0 - y_pad, y1 + y_pad

    def px(value: float) -> float:
        return left + (right - left) * (value - x0) / (x1 - x0)

    def py(value: float) -> float:
        return bottom - (bottom - top) * (value - y0) / (y1 - y0)

    for index in range(6):
        x = left + (right - left) * index / 5
        y = bottom - (bottom - top) * index / 5
        draw.line((x, top, x, bottom), fill=_GRID)
        draw.line((left, y, right, y), fill=_GRID)
        draw.text(
            (x, bottom + 12),
            _number(x0 + (x1 - x0) * index / 5),
            anchor="ma",
            fill=_MUTED,
            font=_font(14),
        )
        draw.text(
            (left - 10, y),
            _number(y0 + (y1 - y0) * index / 5),
            anchor="rm",
            fill=_MUTED,
            font=_font(14),
        )
    largest = sorted(points, key=lambda row: row.size, reverse=True)[:label_top]
    label_ids = {id(row) for row in largest}
    max_size = max(row.size for row in points) or 1.0
    for row in points:
        radius = 5 + 16 * math.sqrt(max(0.0, row.size) / max_size)
        x, y = px(row.x), py(row.y)
        color = _PALETTE[row.series % len(_PALETTE)]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2
        )
        if id(row) in label_ids and row.label:
            draw.text(
                (x + radius + 5, y),
                _short(row.label, 24),
                anchor="lm",
                fill=_INK,
                font=_font(13, bold=True),
            )
    draw.text(((left + right) / 2, 915), x_label, anchor="mm", fill=_MUTED, font=_font(17))
    draw.text((left, top - 18), y_label, anchor="ls", fill=_MUTED, font=_font(17))
    return _png(image)


def quantile_ranges(
    *,
    title: str,
    subtitle: str,
    rows: Sequence[tuple[str, float, float, float, float, float]],
    footnote: str = "",
) -> bytes:
    """Plot label, minimum, q25, median, q75, maximum rows."""

    if not rows:
        raise ValueError("quantile plot requires rows")
    image, draw = _canvas(title, subtitle, footnote)
    left, right, top, bottom = 330, 1500, 180, 850
    low = min(row[1] for row in rows)
    high = max(row[5] for row in rows)
    if low == high:
        high += 1

    def px(value: float) -> float:
        return left + (right - left) * (value - low) / (high - low)

    for index in range(6):
        x = left + (right - left) * index / 5
        value = low + (high - low) * index / 5
        draw.line((x, top, x, bottom), fill=_GRID)
        draw.text((x, bottom + 12), _number(value), anchor="ma", fill=_MUTED, font=_font(14))
    row_height = (bottom - top) / len(rows)
    for index, (label, minimum, q25, median, q75, maximum) in enumerate(rows):
        y = top + row_height * (index + 0.5)
        draw.text((left - 16, y), _short(label, 28), anchor="rm", fill=_INK, font=_font(17))
        draw.line((px(minimum), y, px(maximum), y), fill=_MUTED, width=3)
        draw.rounded_rectangle((px(q25), y - 13, px(q75), y + 13), radius=5, fill="#93C5FD")
        draw.line((px(median), y - 17, px(median), y + 17), fill=_ACCENT, width=5)
        draw.ellipse((px(minimum) - 4, y - 4, px(minimum) + 4, y + 4), fill=_MUTED)
        draw.ellipse((px(maximum) - 4, y - 4, px(maximum) + 4, y + 4), fill=_MUTED)
    return _png(image)
