"""Reviewed complete HS-list owners with variable cardinality, not orphan slots.

Only explicitly declared list blocks may change length. Every block stays on
its original page/span; separators and individual code masks come from that
block. Ordinary scalar HS owners retain their strict fixed-slot contract.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from document_ocr.synthesis.curated import flat
from document_ocr.synthesis.curated_hs import hs_digits, render_source_hs
from document_ocr.synthesis.thermal_goods import render_synthetic_hs_code


@dataclass(frozen=True)
class HsListBlock:
    key: str
    indices: tuple[int, ...]
    masks: tuple[str, ...]
    separators: tuple[str, ...]


def blocks(blueprint) -> tuple[HsListBlock, ...]:
    declarations = blueprint.ownership_data.get("hs_list_blocks")
    if declarations is None:
        return ()
    path = "documentPatch.goodsItemDetails[0].hsCodes"
    goods = blueprint.target["documentPatch"].get("goodsItemDetails", [])
    if len(goods) != 1 or not goods[0].get("hsCodes"):
        raise ValueError("HS-list ownership requires one goods entry with printed codes")
    codes = goods[0]["hsCodes"]
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("HS-list blocks must be an explicit nonempty ordered list")
    result = []
    for item in declarations:
        if set(item) != {"key", "sourceIndices"}:
            raise ValueError("unknown HS-list ownership property")
        key, indices = item["key"], item["sourceIndices"]
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(i) is not int or not 0 <= i < len(codes) for i in indices)
        ):
            raise ValueError("HS-list indices must name printed source codes")
        regions = [r for r in blueprint.regions if r.key == key]
        if len(regions) != 1 or regions[0].curated_key is not None:
            raise ValueError("each HS-list block requires one complete historical region")
        region = regions[0]
        expected = {f"{path}[{i}]" for i in indices}
        if set(region.target_paths) != expected:
            raise ValueError("HS-list block does not own exactly its source code paths")
        masks = tuple(codes[i] for i in indices)
        cursor, separators = 0, []
        for index, mask in enumerate(masks):
            hs_digits(mask)
            start = region.source.find(mask, cursor)
            if start < cursor:
                raise ValueError("HS code is absent from its reviewed list block")
            gap = region.source[cursor:start]
            if index == 0:
                if gap:
                    raise ValueError("HS-list caption must stay outside the code owner")
            elif not re.fullmatch(r"[\s,;]+", gap):
                raise ValueError("HS-list separators contain unowned content")
            else:
                separators.append(gap)
            cursor = start + len(mask)
        if cursor != len(region.source):
            raise ValueError("HS-list block has unowned trailing content")
        result.append(HsListBlock(key, tuple(indices), masks, tuple(separators)))
    indices = [i for block in result for i in block.indices]
    if indices != list(range(len(codes))) or len({b.key for b in result}) != len(result):
        raise ValueError("HS-list blocks must cover every source code once in order")
    other = [
        r
        for r in blueprint.regions
        if r.key not in {b.key for b in result}
        and any(p.startswith(path + "[") for p in r.target_paths)
    ]
    if other:
        raise ValueError("HS-list has overlapping scalar code owners")
    return tuple(result)


def _counts(total: int, owners: tuple[HsListBlock, ...]) -> list[int]:
    if total < len(owners):
        raise ValueError("sampled HS list cannot populate every printed page/block")
    # Keep each original block populated. Allocate the remainder proportionally
    # to source block sizes; code identities themselves are not source-derived.
    weights = [len(b.indices) for b in owners]
    extra, denominator = total - len(owners), sum(weights)
    counts = [1 + extra * weight // denominator for weight in weights]
    order = sorted(range(len(owners)), key=lambda i: (-(extra * weights[i] % denominator), i))
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return counts


def sample(blueprint, hs6_codes, stream) -> list[str]:
    owners = blocks(blueprint)
    if not owners:
        raise ValueError("variable HS sampling needs reviewed whole-list owners")
    identities = list(dict.fromkeys(hs6_codes))
    if any(not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code) for code in identities):
        raise ValueError("variable HS list requires genuine six-digit classification context")
    output, cursor = [], 0
    for owner, count in zip(owners, _counts(len(identities), owners), strict=True):
        for index in range(count):
            mask = owner.masks[index % len(owner.masks)]
            hs6 = identities[cursor]
            code = render_synthetic_hs_code(
                hs6=hs6,
                output_digits=len(hs_digits(mask)),
                stream=stream.derive(f"{owner.key}:{index}:{hs6}"),
            )
            output.append(render_source_hs(mask, code))
            cursor += 1
    return output


def rendering(blueprint, target) -> tuple[dict[str, str], set[str]]:
    owners = blocks(blueprint)
    if not owners:
        return {}, set()
    codes = target["documentPatch"]["goodsItemDetails"][0]["hsCodes"]
    if not codes or any(count != 1 for count in Counter(map(hs_digits, codes)).values()):
        raise ValueError("HS-list targets require distinct complete printed codes")
    output, cursor = {}, 0
    for owner, count in zip(owners, _counts(len(codes), owners), strict=True):
        values = codes[cursor : cursor + count]
        for index, value in enumerate(values):
            mask = owner.masks[index % len(owner.masks)]
            if render_source_hs(mask, hs_digits(value)) != value:
                raise ValueError("HS-list target does not preserve its block's code mask")
        separators = owner.separators or ("\n",)
        output[owner.key] = "".join(
            (separators[(i - 1) % len(separators)] if i else "") + code
            for i, code in enumerate(values)
        )
        cursor += count
    prefix = "documentPatch.goodsItemDetails[0].hsCodes["
    changed = {
        path
        for path in flat(blueprint.target).keys() | flat(target).keys()
        if path.startswith(prefix)
    }
    return output, changed
