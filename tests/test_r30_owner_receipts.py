import copy
import importlib.util
from pathlib import Path

import pytest

from document_ocr.synthesis.curated import digest

spec = importlib.util.spec_from_file_location("r30_owner_receipts", Path(__file__).parents[1] / "scripts/migrations/r30_owner_receipts.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixtures():
    source = b"AAA BBB CCC"
    edits = [{"key": "name", "byteStart": 0, "byteEnd": 3, "before": "AAA", "after": "DDD", "targetPaths": ["name"]}]
    old = {"sourceDocumentId": "doc_x", "joinedRawText": "DDD BBB CCC", "proof": {"sourceSha256": digest(source), "renderedSha256": digest(b"DDD BBB CCC"), "edits": edits}, "target": {"name": "DDD"}}
    new = copy.deepcopy(old)
    new["proof"]["edits"].append({"key": "public", "byteStart": 4, "byteEnd": 7, "before": "BBB", "after": "BBB", "targetPaths": []})
    return old, new


def test_drops_only_new_noop_without_mutation():
    old, new = fixtures(); snapshot = copy.deepcopy(new)
    result, removed = module.discard_added_noops(new, old)
    assert result == old and new == snapshot and len(removed) == 1


def test_retains_existing_noop():
    old, new = fixtures(); old = copy.deepcopy(new)
    assert module.discard_added_noops(new, old) == (new, [])


def test_retains_changed_added_owner():
    old, new = fixtures(); new["proof"]["edits"][1]["after"] = "EEE"
    new["joinedRawText"] = "DDD EEE CCC"
    new["proof"]["renderedSha256"] = digest(new["joinedRawText"].encode())
    assert module.discard_added_noops(new, old) == (new, [])


def test_rejects_wrong_source():
    old, new = fixtures(); new["sourceDocumentId"] = "doc_y"
    with pytest.raises(ValueError, match="source authority"):
        module.discard_added_noops(new, old)


def test_rejects_wrong_retained_text():
    old, new = fixtures(); new["joinedRawText"] = "different"
    with pytest.raises(ValueError):
        module.discard_added_noops(new, old)


def test_unused_noops_remove_only_redundant_entries():
    old, new = fixtures()
    before = copy.deepcopy(new)
    result, removed = module.discard_unused_noops(new, {"overrides": []})
    assert result == old and len(removed) == 1 and new == before


@pytest.mark.parametrize("overlay", [
    {"overrides": [], "extras": [{"fromKey": "public"}]},
    {"overrides": [], "extras": [{"existingOwners": [{"key": "public"}]}]},
    {"overrides": [{"key": "name", "valueRecipe": {"fromKey": "public"}}]},
    {"overrides": [{"key": "public", "valueRecipe": {"literalText": "X"}}]},
    {"overrides": [{"key": "public", "softLineBreaks": True}]},
])
def test_consumed_or_transformed_noops_remain(overlay):
    _, new = fixtures()
    assert module.discard_unused_noops(new, overlay) == (new, [])


def test_target_bearing_noop_remains():
    _, new = fixtures()
    new['proof']['edits'][1]['targetPaths'] = ['field']
    assert module.discard_unused_noops(new, {'overrides': []}) == (new, [])


def test_unused_noop_rejects_inconsistent_bytes():
    _, new = fixtures()
    new['joinedRawText'] = 'invalid'
    with pytest.raises(ValueError):
        module.discard_unused_noops(new, {'overrides': []})
