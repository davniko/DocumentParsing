"""Recover printed casing without changing an already reviewed field's content.

This is not an extractor: field membership, spelling, punctuation and whitespace
are supplied by an existing target. Ambiguous printed case is an explicit error.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator


def case_key(text: str) -> str:
    return unicodedata.normalize("NFC", text.upper())


def character_clusters(text):
    start = 0
    for index in range(1, len(text)):
        if not unicodedata.combining(text[index]):
            yield start, index
            start = index
    if text:
        yield start, len(text)


def case_tokens(text):
    r"""Word/separator spans, keeping combining marks with their base letter.

    Python's ``\w`` excludes combining marks: splitting a printed Turkish i-dot
    at that mark loses the word's case evidence. Keep the common ASCII regex
    path and use the same character clusters as the literal index otherwise.
    """
    if text.isascii():
        for match in re.finditer(r"\w+|\W+", text):
            yield match.start(), match.end(), match[0][0].isalnum() or match[0][0] == "_"
        return
    first, previous, stop = 0, None, 0
    for start, end in character_clusters(text):
        word = text[start].isalnum() or text[start] == "_"
        if previous is not None and word != previous:
            yield first, start, previous
            first = start
        previous, stop = word, end
    if previous is not None:
        yield first, stop, previous


def selected_fields(target: dict) -> Iterator[tuple[str, str]]:
    for role, value in target.get("parties", {}).items():
        entries = enumerate(value) if isinstance(value, list) else [(None, value)]
        for index, party in entries:
            if not isinstance(party, dict):
                continue
            prefix = f"/parties/{role}" + (f"/{index}" if index is not None else "")
            for field in ("name", "addressLine"):
                if party.get(field):
                    yield f"{prefix}/{field}", party[field]
    for index, goods in enumerate(target.get("goodsItemDetails", [])):
        if goods.get("description"):
            yield f"/goodsItemDetails/{index}/description", goods["description"]


class PrintedCaseIndex:
    """Whitespace-insensitive literal index retaining Unicode character origins."""

    def __init__(self, text: str):
        self.text = text
        self._prepared = False
        self._cache: dict[tuple[str, bool], tuple[str, ...]] = {}

    def _prepare(self):
        # Most fields resolve inside a short owned passage. Do not index the
        # entire bill (or a superseded broader scope) unless it is consulted.
        if self._prepared:
            return
        folded, origins, ends = [], [], []
        for index, end in character_clusters(self.text):
            char = self.text[index:end]
            if not char.isspace():
                upper = case_key(char)
                folded.append(upper)
                origins.extend([index] * len(upper))
                ends.extend([end] * len(upper))
        self.folded = "".join(folded)
        self.origins = origins
        self.ends = ends
        self._prepared = True

    def variants(self, value: str, *, word_edges: bool = True) -> tuple[str, ...]:
        key = value, word_edges
        if key in self._cache:
            return self._cache[key]
        self._prepare()
        needle = case_key("".join(value.split()))
        variants = set()
        start = self.folded.find(needle)
        while needle and start >= 0:
            end = start + len(needle)
            a, b = self.origins[start], self.ends[end - 1]
            whole_chars = (start == 0 or self.origins[start - 1] != a) and (
                end == len(self.origins) or self.origins[end] != self.origins[end - 1]
            )
            edges = not word_edges or (
                (not value[0].isalnum() or a == 0 or not self.text[a - 1].isalnum())
                and (not value[-1].isalnum() or b == len(self.text) or not self.text[b].isalnum())
            )
            # An address/name word occurring only inside an email or URL is
            # not a printed occurrence of that field. Full endpoint text can
            # still be retained when it belongs to a reviewed description.
            token_a, token_b = a, b
            while token_a and not self.text[token_a - 1].isspace():
                token_a -= 1
            while token_b < len(self.text) and not self.text[token_b].isspace():
                token_b += 1
            enclosing = self.text[token_a:token_b]
            if ("@" in enclosing or "://" in enclosing) and not ("@" in value or "://" in value):
                edges = False
            if whole_chars and edges:
                output, cursor, previous = [], start, None
                for lo, hi in character_clusters(value):
                    char = value[lo:hi]
                    if char.isspace():
                        output.append(char)
                        continue
                    for position in range(cursor, cursor + len(case_key(char))):
                        origin = self.origins[position]
                        if origin != previous:
                            output.append(self.text[origin : self.ends[position]])
                            previous = origin
                    cursor += len(case_key(char))
                printed = "".join(output)
                # A space inside an expanded letter (S S versus ß) is not a
                # reversible casing change and must not pass as one.
                if case_key(printed) == case_key(value):
                    variants.add(printed)
            start = self.folded.find(needle, start + 1)
        result = tuple(sorted(variants))
        self._cache[key] = result
        return result

    def contextual_variants(self, value: str, span: tuple[int, int]) -> tuple[str, ...] | None:
        """Intersect case evidence from exact neighbouring target phrases.

        A bare IN may occur in many roles, but PACKED IN PLASTIC DRUMS is
        evidence for this occurrence. Disagreeing contexts remain unresolved;
        neither the first occurrence nor the most common spelling wins.
        """
        words = [(a, b) for a, b, word in case_tokens(value) if word]
        current = next(i for i, word in enumerate(words) if word == span)
        possibilities = None
        for left in range(current, -1, -1):
            a, b = words[left][0], words[current][1]
            if not self.variants(value[a:b], word_edges=False):
                break
            right = current
            while right + 1 < len(words) and self.variants(
                value[a : words[right + 1][1]], word_edges=False
            ):
                right += 1
            if left == current == right:
                continue
            b = words[right][1]
            cases = set()
            prefix = case_key("".join(value[a : span[0]].split()))
            token = value[span[0] : span[1]]
            for phrase in self.variants(value[a:b], word_edges=False):
                aligned = PrintedCaseIndex(phrase)
                aligned._prepare()
                begin, end = len(prefix), len(prefix) + len(case_key(token))
                first, last = aligned.origins[begin], aligned.ends[end - 1]
                part = phrase[first:last]
                if case_key(part) == case_key(token):
                    cases.add(part)
            possibilities = cases if possibilities is None else possibilities & cases
        return tuple(sorted(possibilities)) if possibilities is not None else None


def recover_case(value: str, index: PrintedCaseIndex, *, owned: PrintedCaseIndex | None = None):
    """Return case and its evidence method, or a structured unresolved result.

    Explicitly owned text takes precedence over document-wide repeated wording.
    Fragmented targets can use token-case consensus, never a majority vote or
    first-match guess. All ambiguous tokens are returned for adjudication.
    """
    for method, scope in (
        (("owned_literal", owned),) if owned is not None else (("input_literal", index),)
    ):
        variants = scope.variants(value, word_edges=False)
        if len(variants) == 1:
            return {"value": variants[0], "method": method}
        if len(variants) > 1:
            return {"unresolved": [{"text": value, "variants": list(variants)}], "method": method}
    output, unresolved, methods = [], [], set()
    for start, stop, _ in case_tokens(value):
        token = value[start:stop]
        if not token or token.isspace() or token.lower() == token.upper():
            output.append(token)
            continue
        variants = owned.variants(token, word_edges=True) if owned is not None else ()
        method = "owned_token_consensus"
        found_owned = bool(variants)
        if not variants:
            variants = index.variants(token, word_edges=True)
            method = "input_token_consensus"
        if len(variants) != 1:
            contexts = owned.contextual_variants(value, (start, stop)) if owned is not None else None
            method = "owned_context_consensus"
            if contexts is None and not found_owned:
                contexts = index.contextual_variants(value, (start, stop))
                method = "input_context_consensus"
            if contexts is not None:
                variants = contexts
        if len(variants) != 1:
            unresolved.append({"text": token, "variants": list(variants)})
        else:
            output.append(variants[0])
            methods.add(method)
    if unresolved:
        return {"unresolved": unresolved, "method": "token_consensus"}
    printed = "".join(output)
    if case_key(printed) != case_key(value):
        raise ValueError("case recovery changed field content")
    return {"value": printed, "method": "+".join(sorted(methods)) or "no_cased_characters"}
