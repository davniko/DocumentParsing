from __future__ import annotations

from raw_text_template_experiment.selection import carrier_resolution_classification


def test_source_label_carrier_is_immediately_eligible() -> None:
    assert (
        carrier_resolution_classification(
            source={
                "joinedRawText": "no carrier text required",
                "target": {"documentPatch": {"parties": {"carrier": {"name": "Carrier Ltd."}}}},
            },
        )
        == "source_label_present"
    )


def test_vessel_master_source_label_is_not_a_reusable_carrier() -> None:
    source = {
        "joinedRawText": "Signature\nFOR AND ON BEHALF OF MASTER OF MV AGNES\n",
        "target": {
            "documentPatch": {
                "parties": {"carrier": {"name": "MASTER OF MV AGNES"}},
                "transport": {"vesselName": "MV AGNES"},
            }
        },
    }
    assert (
        carrier_resolution_classification(source=source)
        == "source_label_vessel_master_not_reusable_carrier"
    )


def test_country_role_leakage_is_not_a_carrier_principal() -> None:
    source = {
        "joinedRawText": "For the Carrier: Agent for The Carrier - Spain -\n",
        "target": {
            "documentPatch": {
                "parties": {
                    "carrier": {"name": "Spain"},
                    "shipper": {"name": "Exporter Ltd.", "country": "SPAIN"},
                }
            }
        },
    }
    assert (
        carrier_resolution_classification(source=source)
        == "source_label_country_role_not_carrier_principal"
    )


def test_explicit_carrier_placeholders_require_external_enrichment() -> None:
    for carrier_name in ("NA", "Name A/S", "THE CARRIER"):
        source = {
            "joinedRawText": f"Signed for the Carrier: {carrier_name}\n",
            "target": {"documentPatch": {"parties": {"carrier": {"name": carrier_name}}}},
        }
        assert (
            carrier_resolution_classification(source=source)
            == "source_label_placeholder_not_carrier_principal"
        )


def test_master_of_vessel_signature_does_not_establish_a_carrier_principal() -> None:
    source = {
        "joinedRawText": (
            "--- PAGE 1 ---\nSignature\nFOR AND ON BEHALF OF THE\n"
            "MASTER OF M/V NORTH STAR\nCAPT. EXAMPLE\n"
        )
    }
    assert (
        carrier_resolution_classification(source=source)
        == "missing_unresolvable_without_external_enrichment"
    )


def test_generic_or_non_signature_carrier_language_is_not_spend_eligible() -> None:
    samples = (
        "--- PAGE 1 ---\nAS CARRIER / AGENTS FOR THE CARRIER\n",
        "--- PAGE 1 ---\nDelivery agent: Example Logistics Ltd.\nAs Carrier:\n",
        "--- PAGE 1 ---\nCargo addressed to THE MASTER OF M/V NORTH STAR\n",
    )
    for raw in samples:
        assert (
            carrier_resolution_classification(source={"joinedRawText": raw})
            == "missing_unresolvable_without_external_enrichment"
        )
