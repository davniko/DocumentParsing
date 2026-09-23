from __future__ import annotations

import random
import string
from concurrent.futures import ThreadPoolExecutor

import pytest

from document_ocr.synthesis.raw_text_template import (
    build_template_slot,
    compile_raw_text_template,
    printed_topology_mismatches,
    render_compiled_template,
    sentinel_bindings,
    validate_slot_replacements,
)


def test_printed_topology_ignores_relation_metadata_but_rejects_new_printed_slots() -> None:
    source = {
        "schemaVersion": "3",
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "FISH", "additionalInformation": ["GRADE A"]}
            ],
            "cargoPackages": [
                {"groupId": "g1", "packageId": "p1", "quantity": 4, "typeCategory": "BOX"}
            ],
            "containers": [{"containerNumber": "MSCU1234567"}],
        },
    }
    compatible = {
        "schemaVersion": "5",
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "other", "description": "PEARS", "additionalInformation": ["GRADE B"]}
            ],
            "cargoPackages": [
                {
                    "groupId": "other",
                    "packageId": "other-package",
                    "quantity": 9,
                    "typeCategory": "PACKAGE_CARTON",
                }
            ],
            "containers": [
                {
                    "containerNumber": "TLLU7654321",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ],
        },
    }
    assert printed_topology_mismatches(source, compatible) == ()

    incompatible = {
        **compatible,
        "documentPatch": {
            **compatible["documentPatch"],
            "containers": [
                {
                    **compatible["documentPatch"]["containers"][0],
                    "temperatureSetpoint": {"value": -18.0, "unit": "celsius"},
                }
            ],
        },
    }
    mismatches = printed_topology_mismatches(source, incompatible)
    assert {(row.path, row.source_count, row.target_count) for row in mismatches} == {
        ("documentPatch.containers[].temperatureSetpoint.unit", 0, 1),
        ("documentPatch.containers[].temperatureSetpoint.value", 0, 1),
    }


def _slot(
    source: bytes,
    text: str,
    *,
    slot_id: str = "slot_0001",
    occurrence: int = 0,
    render_policy: str = "natural_text",
):
    encoded = text.encode("utf-8")
    cursor = 0
    start = -1
    for _ in range(occurrence + 1):
        start = source.index(encoded, cursor)
        cursor = start + len(encoded)
    return build_template_slot(
        slot_id=slot_id,
        byte_start=start,
        byte_end=start + len(encoded),
        source_text=text,
        target_paths=("documentPatch.test",),
        semantic_role="test",
        evidence_origin="accepted_label_evidence",
        render_policy=render_policy,
    )


def test_copied_invalid_slot_id_is_rejected_at_the_compiler_boundary() -> None:
    source = b"ALPHA"
    valid = _slot(source, "ALPHA")
    copied = valid.model_copy(update={"slot_id": "slot_100000"})
    with pytest.raises(ValueError, match="four-digit slot namespace"):
        compile_raw_text_template(document_id="example", source=source, slots=(copied,))


def test_template_round_trips_utf8_crlf_and_preserves_literal_bytes() -> None:
    source = "--- PAGE 1 ---\r\nSHIPPER\r\n  Ångström GmbH  \r\nKEEP—EXACT\r\n".encode()
    slot = _slot(source, "  Ångström GmbH  ")

    template = compile_raw_text_template(document_id="doc_test", source=source, slots=(slot,))
    round_trip, round_trip_proof = render_compiled_template(
        source=source,
        template=template,
        bindings={slot.slot_id: slot.source_text},
    )
    rendered, proof = render_compiled_template(
        source=source,
        template=template,
        bindings={slot.slot_id: "  Meridian GmbH  "},
    )

    assert round_trip == source
    assert round_trip_proof.source_round_trip is True
    assert rendered == source.replace("Ångström GmbH".encode(), b"Meridian GmbH")
    assert proof.exact_literal_regions is True
    assert proof.page_markers_unchanged is True
    assert proof.line_endings_preserved is True


def test_identifier_slot_requires_exact_character_and_punctuation_shape() -> None:
    source = b"--- PAGE 1 ---\nBOOKING: AB-123/ZX\n"
    slot = _slot(source, "AB-123/ZX", render_policy="opaque_identifier")
    template = compile_raw_text_template(document_id="doc_test", source=source, slots=(slot,))

    rendered, _proof = render_compiled_template(
        source=source,
        template=template,
        bindings={slot.slot_id: "CD-987/QP"},
    )
    assert rendered == b"--- PAGE 1 ---\nBOOKING: CD-987/QP\n"
    with pytest.raises(ValueError, match="identifier shape"):
        render_compiled_template(
            source=source,
            template=template,
            bindings={slot.slot_id: "CD987-QP"},
        )


def test_template_refuses_overlap_source_drift_and_incomplete_bindings() -> None:
    source = b"--- PAGE 1 ---\nALPHA BETA\n"
    alpha = _slot(source, "ALPHA", slot_id="slot_0001")
    overlapping = build_template_slot(
        slot_id="slot_0002",
        byte_start=source.index(b"ALPHA") + 2,
        byte_end=source.index(b"ALPHA") + 8,
        source_text="PHA BE",
        target_paths=("documentPatch.other",),
        semantic_role="other",
        evidence_origin="accepted_label_evidence",
        render_policy="natural_text",
    )
    with pytest.raises(ValueError, match="overlap"):
        compile_raw_text_template(
            document_id="doc_test", source=source, slots=(alpha, overlapping)
        )

    template = compile_raw_text_template(document_id="doc_test", source=source, slots=(alpha,))
    with pytest.raises(ValueError, match="pinned source"):
        render_compiled_template(
            source=source.replace(b"BETA", b"ZETA"),
            template=template,
            bindings={alpha.slot_id: "OMEGA"},
        )
    with pytest.raises(ValueError, match="bindings differ"):
        render_compiled_template(source=source, template=template, bindings={})


def test_template_refuses_line_case_and_edge_whitespace_drift() -> None:
    source = b"--- PAGE 1 ---\n  OLD PARTY  \n"
    slot = _slot(source, "  OLD PARTY  ")
    template = compile_raw_text_template(document_id="doc_test", source=source, slots=(slot,))

    for replacement, message in (
        ("OLD PARTY", "leading whitespace"),
        ("  NEW PARTY\nSECOND  ", "line endings"),
        ("  New Party  ", "case profile"),
    ):
        with pytest.raises(ValueError, match=message):
            render_compiled_template(
                source=source,
                template=template,
                bindings={slot.slot_id: replacement},
            )


def test_partial_slot_validation_uses_the_renderers_exact_format_contract() -> None:
    source = b"--- PAGE 1 ---\n  OLD PARTY  \nREFERENCE: AB-123/ZX\n"
    party = _slot(source, "  OLD PARTY  ", slot_id="slot_0001")
    reference = _slot(
        source,
        "AB-123/ZX",
        slot_id="slot_0002",
        render_policy="opaque_identifier",
    )
    template = compile_raw_text_template(
        document_id="doc_test",
        source=source,
        slots=(party, reference),
    )

    validate_slot_replacements(
        template=template,
        replacements={party.slot_id: "  NEW PARTY  "},
    )
    with pytest.raises(ValueError, match="identifier shape"):
        validate_slot_replacements(
            template=template,
            replacements={reference.slot_id: "CD987-QP"},
        )
    with pytest.raises(ValueError, match="unknown template slots"):
        validate_slot_replacements(template=template, replacements={"slot_9999": "VALUE"})


def test_sentinel_render_changes_only_slots() -> None:
    source = b"--- PAGE 1 ---\nALPHA\nKEEP\nBETA\n"
    slots = (
        _slot(source, "ALPHA", slot_id="slot_0001"),
        _slot(source, "BETA", slot_id="slot_0002"),
    )
    template = compile_raw_text_template(document_id="doc_test", source=source, slots=slots)
    rendered, proof = render_compiled_template(
        source=source,
        template=template,
        bindings=sentinel_bindings(template),
        validate_format=False,
    )

    assert rendered == b"--- PAGE 1 ---\nXXXXX\nKEEP\nXXXX\n"
    assert proof.exact_literal_regions is True
    assert proof.page_markers_unchanged is True
    assert proof.line_endings_preserved is True


def test_sentinel_supports_punctuation_only_evidence() -> None:
    source = b"--- PAGE 1 ---\nVALUE: ---\n"
    slot = _slot(source, "---", occurrence=2)
    template = compile_raw_text_template(document_id="doc_test", source=source, slots=(slot,))

    rendered, proof = render_compiled_template(
        source=source,
        template=template,
        bindings=sentinel_bindings(template),
        validate_format=False,
    )

    assert rendered == b"--- PAGE 1 ---\nVALUE: ---X\n"
    assert proof.exact_literal_regions is True


def test_property_style_randomized_round_trip_and_concurrent_determinism() -> None:
    rng = random.Random(20260905)
    alphabet = string.ascii_letters + string.digits + " .,:;/_-ÅÉ"
    cases = []
    for case_number in range(250):
        left = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 80)))
        value = "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.randrange(2, 25)))
        right = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 80)))
        source = f"--- PAGE 1 ---\n{left}\n{value}\n{right}\n".encode()
        slot = _slot(source, value)
        template = compile_raw_text_template(
            document_id=f"doc_{case_number}", source=source, slots=(slot,)
        )
        cases.append((source, slot, template))

    def render(case):
        source, slot, template = case
        return render_compiled_template(
            source=source,
            template=template,
            bindings={slot.slot_id: slot.source_text},
        )[0]

    sequential = [render(case) for case in cases]
    with ThreadPoolExecutor(max_workers=16) as executor:
        concurrent = list(executor.map(render, cases))

    assert sequential == concurrent
    assert sequential == [source for source, _slot_value, _template in cases]
