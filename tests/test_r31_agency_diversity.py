"""Scoped dataset repair guards; source wording does not dictate company suffixes."""
import json

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.spatial_inputs.paddle import pages_from_record, serialize, text_view_digests
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog
from scripts.analysis.r31_boundary_quality import boundary_readback
from scripts.migrations.diversify_r31_agency_suffixes import ROOT, OUT, check_scoped_change


def read(path):
    return json.loads((ROOT / path).read_text())


@pytest.fixture(scope='module')
def catalog():
    return PaddleTemplateCatalog(ROOT, read(OUT / 'staged.json')['catalog'])


PLAN = read(OUT / 'plan.json')['changes']


@pytest.mark.parametrize('change', PLAN, ids=[r['id'] for r in PLAN])
def test_reviewed_suffix_replays_with_preserved_targets_and_boundary(change, catalog):
    folder = OUT / 'records' / change['id']
    old, new = read(folder / 'before.json'), read(folder / 'after.json')
    changed = check_scoped_change(old, new, change['before'], change['after'])
    assert len(changed) == 1
    e = new['evidence']
    for key in ['candidate', 'catalog', 'wording', 'rendered', 'review', 'previousRecord']:
        assert sha256_file(ROOT / e[key]['path']) == e[key]['sha256']
    candidate_path = ROOT / e['candidate']['path']
    wording = read(e['wording']['path'])
    rendered = catalog.render(candidate_path, wording_receipt=wording)
    assert rendered == read(e['rendered']['path'])
    for k in ['target', 'pages', 'positionedText', 'joinedRawText', 'bboxText']:
        assert rendered[k] == new[k]
    pages = pages_from_record(new)
    for mode, field in [('none', 'joinedRawText'), ('centroid', 'positionedText'), ('bbox', 'bboxText')]:
        assert serialize(pages, geometry=mode) == new[field]
    assert all(new[k] == v for k, v in text_view_digests(new).items())
    template = catalog._template(new['sourceDocumentId'])[0]
    boundary = next(b for b in boundary_readback(rendered, read(candidate_path), template['overlay'])
                    if b['path'] == '/parties/deliveryAgent')
    assert boundary['status'] == 'matched'
    assert any(o['kind'] == 'continuation_shared' for o in boundary['occurrences'])
    old_wording = read(old['evidence']['wording']['path'])
    expected = dict(old_wording['values'], delivery_name=change['after'])
    assert wording['values'] == expected
    assert 'AGENCY is permitted but not required' in catalog.wording_specification(candidate_path)['requested']['delivery_name']


@pytest.mark.parametrize('defect', ['address', 'unrelated_line', 'line_count'])
def test_scope_guard_rejects_unrelated_edits(defect):
    change = PLAN[0];folder = OUT / 'records' / change['id']
    old, new = read(folder / 'before.json'), read(folder / 'after.json')
    if defect == 'address':
        new['target']['parties']['deliveryAgent']['addressLine'] += ' corrupted'
    elif defect == 'unrelated_line':
        new['pages'][0]['regions'][0]['text'] += ' corrupted'
    else:
        new['pages'][0]['regions'].pop()
    with pytest.raises(AssertionError):
        check_scoped_change(old, new, change['before'], change['after'])
