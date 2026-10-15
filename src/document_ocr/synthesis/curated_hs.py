"""Source-shaped tariff identifiers, separate from genuine HS6 commodity identity."""

from __future__ import annotations

import re
from collections.abc import Sequence

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.thermal_goods import render_synthetic_hs_code

_CODE = re.compile(r"[0-9]{6,20}")
_SURFACE = re.compile(r"[0-9][0-9.,:\- \t\r\n]*[0-9]")


def hs_digits(surface: str) -> str:
    """Accept only a complete owned code, never a caption or unrelated number."""
    if not _SURFACE.fullmatch(surface):
        raise ValueError(f"unsupported owned HS presentation: {surface!r}")
    digits = re.sub(r"[^0-9]", "", surface)
    if not _CODE.fullmatch(digits):
        raise ValueError("owned HS presentation must contain 6-20 ASCII digits")
    return digits


def render_source_hs(surface: str, code: str) -> str:
    """Replace digits once, preserving every source separator and its position."""
    original = hs_digits(surface)
    if not _CODE.fullmatch(code) or len(code) != len(original):
        raise ValueError("sampled HS code must match its source presentation width")
    digits = iter(code)
    return "".join(next(digits) if c in "0123456789" else c for c in surface)


def sample_source_hs_codes(
    source_codes: Sequence[str], hs6_codes: Sequence[str], stream: DeterministicStream
) -> list[str]:
    """Preserve printed code slots while sampling their commodity identities.

    Several national extensions may share one HS6 identity. A campaign may also
    explicitly select one identity per printed slot. Suffixes are fictional
    parsing examples, not claims of valid national tariff classifications.
    """
    source_codes = [hs_digits(code) for code in source_codes]
    if any(not re.fullmatch(r"[0-9]{6}", c) for c in hs6_codes):
        raise ValueError("commodity identities must be genuine six-digit HS codes")
    if len(set(source_codes)) != len(source_codes) or len(set(hs6_codes)) != len(hs6_codes):
        raise ValueError("source codes and commodity identities must be distinct")
    parents = list(dict.fromkeys(c[:6] for c in source_codes))
    if len(hs6_codes) == len(source_codes):
        mapped = list(hs6_codes)
    elif len(hs6_codes) == len(parents):
        mapped = [hs6_codes[parents.index(c[:6])] for c in source_codes]
    else:
        raise ValueError("sampled HS identities do not cover the source code slots or HS6 groups")
    output: list[str] = []
    for source, hs6 in zip(source_codes, mapped, strict=True):
        # Independent original codes remain independent even when their HS6
        # prefixes coincide. Repeated printed aliases reuse the resulting code.
        for attempt in range(10 ** (len(source) - 6)):
            code = render_synthetic_hs_code(
                hs6=hs6,
                output_digits=len(source),
                stream=stream.derive(f"{source}:{hs6}:{attempt}"),
            )
            if code not in output:
                output.append(code)
                break
        else:
            raise ValueError("distinct source HS slots exhausted the synthetic suffix domain")
    return output
