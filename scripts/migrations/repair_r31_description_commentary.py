"""Remove two manually reviewed generation disclaimers, with complete replay."""
import copy
import json
import math
import os
import tempfile
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.spatial_inputs.paddle import text_view_digests
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog
from document_ocr.synthesis.scoped_cargo_refresh import apply_edits, reconstruct_source

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/curated/mpci-bl-r31-paddle-centroids-v1'
OUT = ROOT / 'artifacts/synthesis-campaigns/r31-final-readiness-20261109'
DECISIONS = {
    'syn_full_v7_44c7deaad18b67c95db8d2f0': ('v008', 'FICTIONAL PRODUCER', 'PRODUCER'),
    'syn_full_v7_c8a29e30e486d96da2e198d5': ('v023', '; FICTIONAL BRAND NAMES, NOT CONTAINING PROTECTED MARKS', ''),
}


def read(path):
    return json.loads((ROOT / path).read_text())


def pin(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(path)}


def repair(candidate, key, before, after):
    result = copy.deepcopy(candidate)
    edits = [e for e in result['proof']['edits'] if e['key'] == key]
    assert edits and all(e['targetPaths'] == ['documentPatch.goodsItemDetails[0].description'] for e in edits)
    assert all(e['after'].count(before) == 1 for e in edits)
    for e in edits:
        e['after'] = e['after'].replace(before, after)
    for field in ['renderValues', 'surfaceValues']:
        if key in result[field]:
            assert result[field][key].count(before) == 1
            result[field][key] = result[field][key].replace(before, after)
    g = result['target']['documentPatch']['goodsItemDetails']
    assert len(g) == 1 and g[0]['description'].count(before) == 1
    g[0]['description'] = g[0]['description'].replace(before, after)
    source = reconstruct_source(candidate)
    raw = apply_edits(source, result['proof']['edits']).decode()
    result['joinedRawText'] = raw
    result['joinedRawTextSha256'] = result['proof']['renderedSha256'] = digest(raw.encode())
    assert reconstruct_source(result) == source
    assert result['scenario'] == candidate['scenario'] and result['physical'] == candidate['physical']
    return result


def check_reflow(old_render, rendered, key, before, after):
    """Check the lexical scope and account for normal downstream row movement."""
    def blocks(record):
        return {(p['page'], b['owner']): b
                for p in record['geometry']['pages'] for b in p['blocks']}
    old_blocks, new_blocks = blocks(old_render), blocks(rendered)
    assert old_blocks.keys() == new_blocks.keys()
    owned = {(s['page'], s['region']) for proof in old_render['proof']
             if proof['owner'].startswith(key+'@') for s in proof['spans']}
    assert owned
    changed_text, shifted = [], []
    for block, old in old_blocks.items():
        new = new_blocks[block]
        if old == new:
            continue
        if old['text'] != new['text']:
            assert any((block[0], i) in owned for i in old['sourceRegions'])
            assert old['text'].count(before) == 1
            assert new['text'] == old['text'].replace(before, after)
            changed_text.append(block)
            continue
        # Shortening a wrapped goods block moves dependent rows upward. Preserve
        # their text, horizontal placement, dimensions and relative geometry.
        assert {k:v for k,v in old.items() if k != 'physicalFragments'} == {
            k:v for k,v in new.items() if k != 'physicalFragments'}
        offsets = []
        for a, b in zip(old['physicalFragments'], new['physicalFragments'], strict=True):
            assert a['text'] == b['text']
            dx, dy, dr, db = [v-u for u,v in zip(a['box'], b['box'], strict=True)]
            assert abs(dx) < 1e-7 and abs(dr) < 1e-7
            assert math.isclose(dy, db, abs_tol=1e-7) and dy <= 0
            offsets.append(dy)
        assert offsets and all(math.isclose(d, offsets[0], abs_tol=1e-7) for d in offsets)
        shifted.append({'block': block, 'verticalOffsetPixels': offsets[0]})
    assert changed_text
    for moved in shifted:
        page = moved['block'][0]
        products = [b for b in changed_text if b[0] == page]
        assert products
        # Movement cannot exceed the actual removed physical height.
        shrink = max(max(f['box'][3] for f in old_blocks[b]['physicalFragments']) -
                     max(f['box'][3] for f in new_blocks[b]['physicalFragments']) for b in products)
        assert abs(moved['verticalOffsetPixels']) <= shrink + 1e-7
        assert min(f['box'][1] for f in old_blocks[moved['block']]['physicalFragments']) >= min(
            f['box'][1] for b in products for f in old_blocks[b]['physicalFragments'])
    for a, b in zip(old_render['geometry']['pages'], rendered['geometry']['pages'], strict=True):
        assert a['verticalScale'] == b['verticalScale'] and not b['geometryFailures']
    return changed_text, shifted


def main():
    manifest = read(DATA / 'manifest.json')
    assert 'readinessDescriptionAmendment' not in manifest
    aggregate = DATA / 'synthetic-train.jsonl'
    assert pin(aggregate) == manifest['aggregateRecords']['synthetic-train']
    native = PaddleTemplateCatalog(ROOT, manifest['templateCatalog'])
    amendments = []
    for line in aggregate.open():
        row = json.loads(line)
        if row['documentId'] not in DECISIONS:
            continue
        key, before, after = DECISIONS[row['documentId']]
        folder = OUT / row['documentId']
        atomic_write_json(folder / 'before.json', row)
        e = row['evidence']; original = read(e['candidate']['path']); wording = read(e['wording']['path'])
        old_render = native.render(ROOT / e['candidate']['path'], wording_receipt=wording)
        for field in ['pages', 'target', 'positionedText', 'joinedRawText', 'bboxText']:
            assert old_render[field] == row[field], 'baseline replay differs'
        candidate = repair(original, key, before, after)
        lexical = {'status': 'MAIN_AGENT_REVIEWED_LEXICAL_CORRECTION', 'values': candidate['renderValues'],
                   'previousWordingReceiptSha256': original['wordingReceiptSha256'],
                   'previousCandidate': e['candidate'], 'owner': key, 'before': before, 'after': after,
                   'reason': 'Remove drafting commentary; preserve every commercial product fact and shipment value.'}
        atomic_write_json(folder / 'lexical-wording.json', lexical)
        candidate['wordingReceiptSha256'] = digest(lexical)
        candidate['readinessDescriptionRepair'] = pin(folder / 'lexical-wording.json')
        atomic_write_json(folder / 'candidate.json', candidate)
        wording['reviewedCandidateRebase'] = {'beforeSha256': wording['candidateSha256'],
                                             'reason': 'Only product-generation commentary removed; all native extra values remain unchanged.'}
        wording['candidateSha256'] = sha256_file(folder / 'candidate.json')
        atomic_write_json(folder / 'native-wording.json', wording)
        rendered = native.render(folder / 'candidate.json', wording_receipt=wording)
        expected = copy.deepcopy(row['target'])
        assert expected['goodsItemDetails'][0]['description'].count(before) == 1
        expected['goodsItemDetails'][0]['description'] = expected['goodsItemDetails'][0]['description'].replace(before, after)
        assert rendered['target'] == expected, 'other target changed'
        changed, shifted = check_reflow(old_render, rendered, key, before, after)
        atomic_write_json(folder / 'rendered.json', rendered)
        review = {'status': 'PASS', 'before': before, 'after': after, 'priorEvidence': e,
                  'changedTextBlocks': changed, 'translatedBlocks': shifted,
                  'otherTargetsAndTextUnchanged': True,
                  'geometry': 'Normal measured reflow; unrelated rows may translate vertically without changing size or horizontal position.',
                  'lexicalWording': pin(folder / 'lexical-wording.json'), 'apiCalls': 0}
        atomic_write_json(folder / 'review.json', review)
        updated = copy.deepcopy(row)
        updated.update({k: rendered[k] for k in ['target','pages','joinedRawText','positionedText','bboxText']})
        updated.update(text_view_digests(updated))
        updated['evidence'] = {'candidate': pin(folder / 'candidate.json'), 'wording': pin(folder / 'native-wording.json'),
                               'rendered': pin(folder / 'rendered.json'), 'catalog': manifest['templateCatalog'],
                               'review': pin(folder / 'review.json'), 'previousRecord': pin(folder / 'before.json')}
        updated['readinessDescriptionAmendment'] = pin(folder / 'review.json')
        atomic_write_json(folder / 'after.json', updated)
        amendments.append({'id': row['documentId'], 'before': pin(folder / 'before.json'),
                           'after': pin(folder / 'after.json'), 'review': pin(folder / 'review.json')})
        print(row['documentId'], 'PASS', changed, flush=True)
    assert len(amendments) == 2
    atomic_write_json(OUT / 'manifest-before.json', manifest)
    replacements = {r['id']: read(r['after']['path']) for r in amendments}
    with tempfile.NamedTemporaryFile(dir=DATA, prefix='.readiness-', delete=False) as stream:
        temporary = Path(stream.name)
        for line in aggregate.open('rb'):
            row = json.loads(line)
            if row['documentId'] in replacements:
                line = (json.dumps(replacements[row['documentId']], ensure_ascii=False, separators=(',', ':'))+'\n').encode()
            stream.write(line)
        stream.flush(); os.fsync(stream.fileno())
    receipt = {'status': 'PASS', 'changes': amendments, 'beforeManifest': pin(DATA / 'manifest.json'),
               'before': manifest['aggregateRecords']['synthetic-train'],
               'after': {'path': str(aggregate.relative_to(ROOT)), 'sha256': sha256_file(temporary)},
               'changedRecords': 2, 'otherSyntheticRealValidationUnchanged': True, 'apiCalls': 0}
    atomic_write_json(OUT / 'receipt.json', receipt)
    os.replace(temporary, aggregate)
    manifest['aggregateRecords']['synthetic-train'] = receipt['after']
    manifest['readinessDescriptionAmendment'] = pin(OUT / 'receipt.json')
    manifest['unchanged'].append('Final readiness amendment removes generation commentary from two older description owners; all other target values and text remain unchanged. Normal measured reflow translates dependent rows where needed; all geometry is regenerated and validated. Prior stages and receipts remain immutable.')
    atomic_write_json(DATA / 'manifest.json', manifest)
    print(json.dumps({'aggregate': receipt['after'], 'manifest': pin(DATA / 'manifest.json')}))


if __name__ == '__main__':
    main()
