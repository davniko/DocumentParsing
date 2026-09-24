from __future__ import annotations

from document_ocr.synthesis.template_integrity import (
    explicit_carrier_receipt_container_counts,
    source_template_integrity_issues,
)


def test_carrier_receipt_container_count_accepts_same_and_following_line_grammars() -> None:
    raw = (
        "CARRIER'S RECEIPT (No. of Cntrs or Pkgs rcvd by Carrier)\n"
        "5 CNTRS\n"
        "CARRIER\N{RIGHT SINGLE QUOTATION MARK}S RECEIPT: 3 CONTAINER(S)\n"
        "Cargo contains 99 containers\n"
    )

    assert explicit_carrier_receipt_container_counts(raw) == (5, 3)


def test_carrier_receipt_container_count_accepts_words_parentheses_and_suffixes() -> None:
    raw = (
        "CARRIER'S RECEIPT: Total number of containers received\n"
        "(FOUR) CONTAINER(S) ONLY\n"
        "CARRIER'S RECEIPT: TWENTY-ONE CONTAINERS\n"
        "CARRIER'S RECEIPT: Total number of containers or packages received\n"
    )

    assert explicit_carrier_receipt_container_counts(raw) == (4, 21)


def test_carrier_receipt_container_count_rejects_invalid_word_sequences() -> None:
    raw = "CARRIER'S RECEIPT\nONE ONE CONTAINERS\n"

    assert explicit_carrier_receipt_container_counts(raw) == ()


def test_explicit_received_and_weight_total_container_fields():
    assert explicit_carrier_receipt_container_counts(
        "Total No. of Containers received by the Carrier: 17\n"
        "Weight in Kgs Total: 10 CONTAINER(S)\n"
    ) == (17, 10)
    assert (
        explicit_carrier_receipt_container_counts(
            "Freight invoice: 17\nWeight in Kgs Total: 100 KG\n"
        )
        == ()
    )


def test_source_template_integrity_rejects_only_a_proven_count_contradiction() -> None:
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "AAAA0000000"},
                {"containerNumber": "BBBB0000000"},
                {"containerNumber": "CCCC0000000"},
            ]
        }
    }

    assert source_template_integrity_issues("CARRIER'S RECEIPT\n5 CNTRS\n", target) == (
        "explicit_carrier_receipt_container_count_differs_from_labeled_containers:5_vs_3",
    )
    assert source_template_integrity_issues("CARRIER'S RECEIPT\n3 CNTRS\n", target) == ()
    assert source_template_integrity_issues("Cargo contains 5 containers\n", target) == ()
    assert source_template_integrity_issues(
        "CARRIER'S RECEIPT\n(FOUR) CONTAINER(S) ONLY\n", target
    ) == ("explicit_carrier_receipt_container_count_differs_from_labeled_containers:4_vs_3",)


def test_explicit_shipment_count_is_independent_of_compiled_binding_ownership():
    target = {"documentPatch": {"containers": [{"containerNumber": "FFAU4350822"}]}}
    raw = "FFAU 4350822/HC40\n02 X 40\u2019HC FCL CONTAINERS SAID TO CONTAIN 6043 PKGS"
    assert source_template_integrity_issues(raw, target) == (
        "explicit_shipment_container_count_exceeds_labeled_containers:2_vs_1",
    )
    target["documentPatch"]["containers"].append({"containerNumber": "AAAA0000000"})
    assert source_template_integrity_issues(raw, target) == ()
    # A subgroup is a lower bound, not a declaration of the whole inventory.
    assert source_template_integrity_issues(
        "1 X 40HC CONTAINER SAID TO CONTAIN 600 PKGS", target
    ) == ()
    assert source_template_integrity_issues(
        "Rate for 99 X 40HC CONTAINERS SAID TO CONTAIN goods: USD 400", target
    ) == ()


def test_source_integrity_requires_container_local_full_seals_and_complete_equipment_codes():
    target = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "FBIU5385937",
                    "typeDescription": "40RF",
                    "sealNumbers": ["96", "WHLW357402"],
                }
            ],
            "cargoGroups": [{"marksAndNumbers": ["501305 / ENOS01721979"]}],
        }
    }
    raw = (
        "FBIU5385937 40RF96 WHLW357402\n"
        "FBIU5385937/HC40\n501305 / ENOS01721979\n"
        "SEAL/ENOS01721621\n"
    )
    assert source_template_integrity_issues(raw, target) == (
        "container_adjacent_seal_pair_not_labeled:0",
        "equipment_type_suffix_mislabeled_as_seal:0",
    )
    target["documentPatch"]["containers"][0].update(
        typeDescription="40RF96",
        sealNumbers=["501305", "ENOS01721979", "WHLW357402"],
    )
    assert source_template_integrity_issues(raw, target) == ()
    target["documentPatch"]["containers"][0]["sealNumbers"][0] = "01721621"
    assert source_template_integrity_issues(raw, target) == (
        "container_adjacent_seal_pair_not_labeled:0",
        "printed_seal_prefix_missing_from_label",
    )
