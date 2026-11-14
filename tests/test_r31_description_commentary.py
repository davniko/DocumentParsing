"""Scope and geometry regression checks for the two reviewed R31 amendments."""
import copy

import pytest

from scripts.migrations.repair_r31_description_commentary import (
    DATA, DECISIONS, OUT, ROOT, PaddleTemplateCatalog, check_reflow, read, repair,
)


@pytest.fixture(scope='module', params=list(DECISIONS))
def replay(request):
    identity = request.param
    folder = OUT / identity
    if not (folder / 'before.json').exists():
        pytest.skip('local amendment evidence is not installed')
    old = read(folder / 'before.json')
    key, before, after = DECISIONS[identity]
    original = read(old['evidence']['candidate']['path'])
    candidate = repair(original, key, before, after)
    assert candidate['scenario'] == original['scenario']
    assert candidate['physical'] == original['physical']
    catalog = PaddleTemplateCatalog(ROOT, read(DATA / 'manifest.json')['templateCatalog'])
    old_render = catalog.render(ROOT / old['evidence']['candidate']['path'],
                                wording_receipt=read(old['evidence']['wording']['path']))
    new_render = catalog.render(folder / 'candidate.json',
                                wording_receipt=read(folder / 'native-wording.json'))
    target = copy.deepcopy(old['target'])
    target['goodsItemDetails'][0]['description'] = target['goodsItemDetails'][0]['description'].replace(before, after)
    assert new_render['target'] == target
    return old_render, new_render, key, before, after


def test_normal_reflow_keeps_all_other_content(replay):
    assert check_reflow(*replay)[0]


def test_rejects_unrelated_text_edit(replay):
    old, new, key, before, after = replay
    new = copy.deepcopy(new)
    changed = check_reflow(*replay)[0]
    block = next(b for p in new['geometry']['pages'] for b in p['blocks']
                 if (p['page'], b['owner']) not in changed)
    block['text'] += ' UNREQUESTED'
    with pytest.raises(AssertionError):
        check_reflow(old, new, key, before, after)


def test_rejects_unrelated_horizontal_shift(replay):
    old, new, key, before, after = replay
    new = copy.deepcopy(new)
    changed = check_reflow(*replay)[0]
    block = next(b for p in new['geometry']['pages'] for b in p['blocks']
                 if (p['page'], b['owner']) not in changed)
    block['physicalFragments'][0]['box'][0] += 1
    with pytest.raises(AssertionError):
        check_reflow(old, new, key, before, after)
