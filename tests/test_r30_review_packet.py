import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/migrations"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("r30_review_packet", SCRIPTS / "r30_review_packet.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
compare = module.compare_page

probe_spec = importlib.util.spec_from_file_location(
    "probe_r30_match_first", SCRIPTS / "probe_r30_match_first.py"
)
probe = importlib.util.module_from_spec(probe_spec)
probe_spec.loader.exec_module(probe)


def test_split_and_joined_lines_keep_evidence():
    result = compare([(0, "STEEL DRUMS"), (1, "224")], [(8, "STEEL DRUMS 224")])
    assert [r["status"] for r in result] == ["literal_unique", "literal_unique"]
    assert result[1]["matches"] == [[[8, 12, 15]]]
    result = compare([(0, "STEEL DRUMS 224")], [(8, "STEEL DRUMS"), (9, "224")])
    assert result[0]["matches"] == [[[8, 0, 11], [9, 0, 3]]]


def test_casing_punctuation_digits_never_normalized():
    for changed in ("5205:22", "520522", "5205.23", "520S.22"):
        assert compare([(0, changed)], [(1, "5205.22")])[0]["status"] == "changed_or_new"
    assert compare([(0, "Drums")], [(1, "DRUMS")])[0]["status"] == "changed_or_new"


def test_extra_occurrence_is_not_hidden():
    assert {r["status"] for r in compare([(0, "BL123"), (1, "BL123")], [(8, "BL123")])} == {
        "competing_occurrence"
    }
    assert compare([(0, "BL123")], [(8, "BL123"), (9, "BL123")])[0]["status"] == "ambiguous"


def test_identifier_substrings_are_not_equal():
    assert compare([(0, "123")], [(8, "BL1234")])[0]["status"] == "changed_or_new"


def test_numeric_punctuation_does_not_hide_a_changed_value():
    for short, long in (("200", "200,000"), ("5", "12.5"), ("000", "200,000")):
        assert compare([(0, short)], [(8, long)])[0]["status"] == "changed_or_new"
    assert compare([(0, "200")], [(8, "200, DRUMS")])[0]["status"] == "literal_unique"


def test_binding_proposals_surface_competing_roles_and_changed_identifiers():
    source = {
        "pages": [
            {
                "index": 0,
                "regions": [
                    {"index": 0, "text": "ALPHA LTD"},
                    {"index": 1, "text": "BL1234"},
                ],
            }
        ]
    }
    owners = [
        {
            "owner": k,
            "key": k,
            "byteStart": i,
            "before": text,
            "status": "unmatched_or_ambiguous_review",
        }
        for i, (k, text) in enumerate(
            [("shipper", "ALPHA LTD"), ("notify", "ALPHALTD"), ("bill", "BL123")]
        )
    ]
    result = module.binding_proposals(source, owners)
    assert [r["status"] for r in result] == [
        "competing_owner_review",
        "competing_owner_review",
        "manual_span_review",
    ]
    assert result[0]["choices"][0][0]["expected"] == "ALPHA LTD"


def test_alternative_shipment_owners_do_not_compete():
    source = {"pages": [{"index": 0, "regions": [{"index": 0, "text": "12 PACKAGES"}]}]}
    owners = [
        {
            "owner": key,
            "key": key,
            "byteStart": 0,
            "before": "12 PACKAGES",
            "status": "unmatched_or_ambiguous_review",
        }
        for key in ("single", "multi")
    ]
    rows = module.binding_proposals(source, owners, [{"single"}, {"multi"}])
    assert all(r["status"] == "unique_span_role_review" for r in rows)


def test_changed_suffix_and_omission_remain_visible():
    left = [(0, "SHIPPER"), (1, "ACME LLC"), (2, "TEL 12345")]
    right = [(8, "SHIPPER"), (9, "ACME LLC"), (10, "TEL 1234")]
    assert compare(left, right)[2]["status"] == "changed_or_new"
    assert compare(right, left)[2]["status"] == "changed_or_new"


def test_order_independent_without_changing_output_order():
    left = [(0, "A LTD"), (1, "B LTD")]
    rows = compare(left, [(8, "B LTD"), (9, "A LTD")])
    assert [r["text"] for r in rows] == ["A LTD", "B LTD"]
    assert rows[0]["matches"] == [[[9, 0, 5]]]


def test_draft_excludes_unresolved_owners_and_preserves_manual_corrections():
    packet = {
        "bindingProposals": [
            {
                "key": "matched",
                "byteStart": 10,
                "status": "unique_span_role_review",
                "choices": [[{"page": 0, "region": 1, "start": 2, "end": 5}]],
            },
            {"key": "ambiguous", "byteStart": 20, "status": "manual_span_review", "choices": []},
            {"key": "conflict", "byteStart": 30, "status": "competing_owner_review", "choices": []},
        ]
    }
    assert probe.draft_overlay(packet, {})["overrides"] == [
        {"key": "matched", "byteStart": 10, "spans": [[0, 1, 2, 5]]}
    ]
    reviewed = {"key": "matched", "byteStart": 10, "spans": [[0, 4, 0, 8]]}
    extra = {"owner": "new:repeat", "fromKey": "matched", "spans": [[0, 5, 0, 8]]}
    result = probe.draft_overlay(packet, {"overrides": [reviewed], "extras": [extra]})
    assert result["overrides"] == [reviewed]
    assert result["extras"] == [extra]
    assert result["targetSets"] == {}
    assert packet["bindingProposals"][0]["choices"][0][0]["region"] == 1


def test_literal_proposal_keeps_full_boundary_and_role_context_for_review():
    source = {
        "pages": [
            {"index": 0, "regions": [{"index": 4, "text": "MUNDRA PORT", "box": [10, 20, 90, 30]}]}
        ]
    }
    owner = {
        "owner": "pol",
        "key": "pol",
        "byteStart": 12,
        "before": "MUNDRA",
        "targetPaths": ["documentPatch.route.portOfLoading.name"],
        "status": "unmatched_or_ambiguous_review",
    }
    proposal = module.binding_proposals(source, [owner])[0]
    assert proposal["status"] == "unique_span_role_review"
    assert proposal["choices"][0][0]["expected"] == "MUNDRA"
    assert proposal["choiceContext"][0][0]["text"] == "MUNDRA PORT"
    assert proposal["choiceContext"][0][0]["box"] == [10, 20, 90, 30]
    assert proposal["targetPaths"] == owner["targetPaths"]
