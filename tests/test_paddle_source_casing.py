import copy

import pytest

from document_ocr.synthesis.paddle_source_casing import project_source_casing, rendered_case_scopes
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog


def fixture():
    box = [0, 0, 100, 20]
    printed = "Acme Ltd, Main Road"
    rendered = {
        "target": {
            "parties": {
                "shipper": {"name": "ACME LTD", "addressLine": "MAIN ROAD", "country": "EGYPT"}
            }
        },
        "pages": [{"regions": []}, {"regions": [{"box": box, "text": printed}]}],
        "proof": [
            {
                "owner": "party@0:4",
                "spans": [{"page": 1, "region": 3, "start": 0, "end": 4, "expected": "OLD"}],
                "after": printed,
            }
        ],
        "geometry": {
            "pages": [
                {
                    "page": 1,
                    "blocks": [
                        {
                            "sourceRegions": [3],
                            "text": printed,
                            "physicalFragments": [{"box": box, "text": printed}],
                        }
                    ],
                }
            ]
        },
    }
    candidate = {
        "proof": {
            "edits": [
                {
                    "key": "party",
                    "byteStart": 0,
                    "byteEnd": 4,
                    "targetPaths": [
                        "documentPatch.parties.shipper.name",
                        "documentPatch.parties.shipper.addressLine",
                    ],
                }
            ]
        }
    }
    return rendered, candidate


def test_projection_uses_explicit_geometry_page_and_does_not_touch_input():
    rendered, candidate = fixture()
    original = copy.deepcopy(rendered)
    result = project_source_casing(rendered, candidate, {}, {})
    assert result["target"]["parties"]["shipper"] == {
        "name": "Acme Ltd",
        "addressLine": "Main Road",
        "country": "EGYPT",
    }
    assert rendered == original
    assert result["pages"] == original["pages"]


def test_casing_rejects_unprinted_geometry():
    rendered, candidate = fixture()
    rendered["pages"][1]["regions"][0]["text"] = "Something else"
    with pytest.raises(ValueError, match="absent from final Paddle"):
        rendered_case_scopes(rendered, candidate, {}, {})


def test_character_owner_disambiguates_merged_primary_and_related_party():
    rendered, candidate = fixture()
    text = "Main Road ON BEHALF OF OTHER LTD MAIN ROAD"
    rendered["target"] = {"parties": {"shipper": {"addressLine": "MAIN ROAD"}}}
    rendered["pages"][1]["regions"][0]["text"] = text
    block = rendered["geometry"]["pages"][0]["blocks"][0]
    block["text"] = block["physicalFragments"][0]["text"] = text
    rendered["proof"][0]["after"] = "Main Road"
    overlay = {
        "targetFromOwners": [
            {
                "target": "/parties/shipper/addressLine",
                "keys": ["party"],
                "separator": " ",
                "targetCase": "upper",
            }
        ]
    }
    assert (
        project_source_casing(rendered, candidate, overlay, {})["target"]["parties"]["shipper"][
            "addressLine"
        ]
        == "Main Road"
    )


def test_unknown_casing_policy_rejected_before_loading_catalog(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "document_ocr.synthesis.paddle_templates.read_pin", lambda *_: {"targetCasing": "guess"}
    )
    with pytest.raises(ValueError, match="unsupported native target casing"):
        PaddleTemplateCatalog(tmp_path, {})


def test_casing_only_validator_rejects_content_and_unrelated_field_changes():
    from scripts.migrations.migrate_r31_casing import validate_case_only

    before = {
        "negotiability": None,
        "issueDate": "2025-05-16",
        "parties": {
            "shipper": {
                "name": "ACME LTD",
                "country": "EGYPT",
                "contactDetails": {"phoneNumbers": ["+20 1234567"]},
            }
        },
    }
    after = copy.deepcopy(before)
    after["parties"]["shipper"]["name"] = "Acme Ltd"
    validate_case_only(before, after)
    after["parties"]["shipper"]["country"] = "Egypt"
    with pytest.raises(ValueError, match="out-of-scope"):
        validate_case_only(before, after)
    after = copy.deepcopy(before)
    after["parties"]["shipper"]["name"] = "Acme"
    with pytest.raises(ValueError, match="changed content"):
        validate_case_only(before, after)


def test_reviewed_care_of_scope_includes_static_caption_from_name_line():
    rendered,candidate=fixture()
    texts=['Broker Co C/O','51230419','Te Maiya Stores','Betio Road','o']
    rendered['target']={'parties':{'shipper':{'addressLine':'C/O TE MAIYA STORES BETIO ROAD'}}}
    rendered['pages'][1]['regions']=[]
    rendered['geometry']['pages'][0]['blocks']=[]
    for region,text in enumerate(texts):
        fragment={'text':text,'box':[0,region*30,100,region*30+20]}
        rendered['pages'][1]['regions'].append(fragment)
        rendered['geometry']['pages'][0]['blocks'].append({'sourceRegions':[region],'text':text,'physicalFragments':[fragment]})
    rendered['proof']=[]
    overlay={'sourceCaseRegions':{'/parties/shipper/addressLine':[[1,0],[1,2],[1,3]]}}
    result=project_source_casing(rendered,candidate,overlay,{})
    assert result['target']['parties']['shipper']['addressLine']=='C/O Te Maiya Stores Betio Road'
    assert result['sourceCasing']['fields'][0]['method']=='owned_literal'
