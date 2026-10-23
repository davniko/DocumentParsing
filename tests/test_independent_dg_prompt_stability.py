"""Native evidence must replay after canonical JSON persistence sorts mappings."""

import json

from document_ocr.synthesis.independent_dg_context import (
    form_prompt,
    form_request,
    hs_prompt,
    hs_request,
)


def test_native_dg_context_prompts_ignore_mapping_insertion_order():
    frozen = {
        "regime": "dg",
        "dangerousGoods": {
            "recordId": "record",
            "unNumber": "1470",
            "properShippingName": "Lead perchlorate, solid",
            "class": "5.1",
            "packingGroup": "II",
            "subsidiaryHazards": ["6.1"],
        },
    }
    persisted = json.loads(json.dumps(frozen, sort_keys=True))
    assert hs_prompt(hs_request(frozen)) == hs_prompt(hs_request(persisted))
    assert form_prompt(form_request(frozen)) == form_prompt(form_request(persisted))
