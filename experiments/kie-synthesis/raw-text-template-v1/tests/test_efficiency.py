from __future__ import annotations

from decimal import Decimal

from raw_text_template_experiment.compact_contract import compact_critic_payload
from raw_text_template_experiment.efficiency import (
    PricingScenario,
    _compact_critic_payload,
    _expand_critic_payload,
    _scenario_cost,
)


def _critic_payload() -> dict[str, object]:
    logical_key = "anchor:documentPatch.billOfLadingNumber"
    return {
        "documentId": "doc_test",
        "allowedTargetPaths": ["documentPatch.billOfLadingNumber"],
        "allowedRemovalLogicalKeys": [logical_key],
        "bindingInventory": [
            {
                "sourceBindingIds": ["anchor_binding_0001"],
                "logicalKey": logical_key,
                "renderMode": "target_binding",
                "valueKind": "identifier",
                "groupKind": "document",
                "groupKey": "document",
                "targetPaths": ["documentPatch.billOfLadingNumber"],
                "targetRelationship": "single_target",
                "independentTargetFactComponents": [["documentPatch.billOfLadingNumber"]],
                "derivation": None,
                "dependencyPaths": [],
                "dependencyBindings": [],
                "occurrences": [
                    {
                        "sourceBindingId": "anchor_binding_0001",
                        "lineStart": "L00003",
                        "lineEnd": "L00003",
                        "sourceText": "ABC123",
                        "occurrenceIndex": 0,
                        "exactMatchCount": 1,
                        "exactMatchCandidates": [],
                    }
                ],
            }
        ],
        "maskedTemplate": f"L00003 | ⟦{logical_key}:target_binding⟧",
    }


def test_compact_critic_payload_round_trips_without_information_loss() -> None:
    original = _critic_payload()

    compact = _compact_critic_payload(original)

    assert compact["maskedTemplate"] == "L00003 | ⟦binding_0000⟧"
    assert compact["occurrenceRows"][0][0] == "occurrence_00000"
    assert _expand_critic_payload(compact) == original


def test_live_compact_contract_retains_audited_tables_and_removal_vocabulary() -> None:
    original = _critic_payload()
    logical_key = "anchor:documentPatch.billOfLadingNumber"
    original["annotatedSource"] = f"L00003 | ⟦{logical_key}:target_binding⟧ABC123⟦/binding⟧"

    compact = compact_critic_payload(original)

    assert compact["compactContract"]["schemaVersion"] == 3
    assert compact["compactContract"]["occurrenceIdFormat"] == "occ_L%05d_%05d"
    assert compact["maskedTemplate"] == "L00003 | ⟦binding_0000⟧"
    assert compact["annotatedSource"] == ("L00003 | ⟦binding_0000⟧ABC123⟦/binding⟧")
    assert compact["targetPathTable"] == [["path_0000", "documentPatch.billOfLadingNumber"]]
    assert compact["bindingRows"][0][0] == "binding_0000"
    assert compact["occurrenceRows"][0][0] == "occ_L00003_00000"
    assert compact["allowedRemovalLogicalKeys"] == ["anchor:documentPatch.billOfLadingNumber"]


def test_same_token_scenario_cost_separates_cache_classes() -> None:
    usage = {
        "inputTokens": 100,
        "cacheReadTokens": 20,
        "cacheWriteTokens": 30,
        "outputTokens": 10,
    }
    scenario = PricingScenario.model_validate(
        {
            "name": "probe",
            "model": "provider/model",
            "source_url": "https://example.test/model",
            "input_usd_per_million": Decimal("1"),
            "cached_input_usd_per_million": Decimal("0.1"),
            "cache_write_multiplier": Decimal("1.25"),
            "output_usd_per_million": Decimal("2"),
        }
    )

    cost = _scenario_cost(usage, scenario)

    assert cost == Decimal("0.0001095")
