"""Reviewable, suffix-only R31 amendment; never rewrite historical receipts."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import resource
import tempfile
import time
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.spatial_inputs.paddle import text_view_digests
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog
from scripts.analysis.r31_boundary_quality import boundary_readback, compact
from scripts.migrations.r31_agency_suffix_review import REVIEW

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/curated/mpci-bl-r31-paddle-centroids-v1'
OUT = ROOT / 'artifacts/synthesis-campaigns/r31-agency-diversity-20261109'
PUBLIC = ('target', 'pages', 'joinedRawText', 'positionedText', 'bboxText')


def read(path):
    return json.loads((ROOT / path).read_text())


def pin(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(path)}


def audit():
    prior = ROOT / 'docs/analysis/r30-training-tweaks-boundaries-20261108'
    rejected = read(prior / 'reviewed.json')['training_screen_rejection']
    rows = [r for r in read(prior / 'nonseparate-cases.json')
            if r['split'] == 'synthetic-train' and r['presentation'].startswith('shared')
            and (r['id'], r['role']) != (rejected['id'], rejected['role'])
            and r['name_anchor']['matched'] < len(compact(r['name']))]
    agency = {r['id']: r for r in rows if re.search(r'AGENCY\W*$', r['name'], re.I)}
    assert len(rows) == 141 and len(agency) == 76
    records, family_counts = [], Counter()
    parents = {r['parent'] for r in agency.values()}
    for index, line in enumerate((DATA / 'synthetic-train.jsonl').open()):
        row = json.loads(line)
        if row['sourceDocumentId'] in parents:
            name = row['target']['parties']['deliveryAgent']['name']
            family_counts[(row['sourceDocumentId'], 'older' if index < 9400 else 'new750',
                           'agency' if re.search(r'AGENCY\W*$', name, re.I) else 'other')] += 1
        if row['documentId'] in agency:
            owner = row['target']['parties']['deliveryAgent']
            receipt = read(row['evidence']['wording']['path'])
            assert receipt['values']['delivery_name'] == owner['name']
            records.append({'id': row['documentId'], 'parent': row['sourceDocumentId'],
                            'name': owner['name'], 'address': owner['addressLine'],
                            'evidence': row['evidence']})
    assert len(records) == 76
    result = {'oldContinuationRoles': len(rows), 'agencyDocuments': len(records),
              'uniqueNames': len({compact(r['name']) for r in records}),
              'casing': dict(Counter('upper' if r['name'].isupper() else 'mixed' for r in records)),
              'qualifiers': dict(Counter(re.search(r'\(([^()]*)\)$', r['name']).group(1).upper()
                                        if re.search(r'\(([^()]*)\)$', r['name']) else 'outside-parentheses'
                                        for r in records)),
              'families': [{'parent': k[0], 'cohort': k[1], 'ending': k[2], 'count': n}
                           for k, n in family_counts.items()], 'records': records}
    atomic_write_json(OUT / 'audit-before.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


def check_scoped_change(old, new, old_name, new_name):
    expected = copy.deepcopy(old['target'])
    assert expected['parties']['deliveryAgent']['name'] == old_name
    expected['parties']['deliveryAgent']['name'] = new_name
    assert new['target'] == expected, 'unrelated target changed'
    assert len(old['pages']) == len(new['pages'])
    changed = []
    for op, np in zip(old['pages'], new['pages'], strict=True):
        assert {k: v for k, v in op.items() if k != 'regions'} == {k: v for k, v in np.items() if k != 'regions'}
        assert len(op['regions']) == len(np['regions']), 'line count changed'
        for index, (a, b) in enumerate(zip(op['regions'], np['regions'], strict=True)):
            if a != b:
                changed.append({'page': op['index'], 'region': index, 'before': a, 'after': b})
    # A suffix may appear more than once, but never authorize an unrelated region.
    old_suffix = re.search(r'Agency\W*$', old_name, re.I).group()
    new_suffix = new_name[len(old_name) - len(old_suffix):]
    for item in changed:
        assert old_suffix in item['before']['text']
        assert item['after']['text'] == item['before']['text'].replace(old_suffix, new_suffix)
    assert changed, 'no rendered suffix changed'
    return changed


def stage():
    started = time.perf_counter()
    manifest = read(DATA / 'manifest.json')
    assert 'agencyDiversityAmendment' not in manifest
    assert pin(DATA / 'synthetic-train.jsonl') == manifest['aggregateRecords']['synthetic-train']
    reviewed = REVIEW
    cases = {r['id']: r for r in read(OUT / 'audit-before.json')['records']}
    plan = {'reviewed': True, 'review': reviewed['review'], 'changes': []}
    for identity, suffix in reviewed['suffixes'].items():
        row = cases[identity]
        before = row['name']
        after, count = re.subn(r'Agency(?=\W*$)', suffix, before, flags=re.I)
        assert count == 1 and before != after
        assert before.isupper() == after.isupper()
        plan['changes'].append({'id': identity, 'parent': row['parent'], 'before': before, 'after': after})
    atomic_write_json(OUT / 'plan.json', plan)
    changes = {r['id']: r for r in plan['changes']}
    assert len(changes) == 40 and plan['reviewed'] is True
    before_catalog = manifest['templateCatalog']
    catalog = read(before_catalog['path'])
    for sid in sorted({r['parent'] for r in plan['changes']}):
        template = read(catalog['templates'][sid]['template']['path'])
        text = template['overlay']['newWording']['delivery_name']
        case = 'uppercase' if 'uppercase' in text else 'title-case'
        template['overlay']['newWording']['delivery_name'] = (
            f'One {case} fictional destination-country delivery/shipping company name with a '
            'parenthesized trading-name or branch qualifier. Vary natural company wording and '
            'qualifier endings; AGENCY is permitted but not required. Do not systematically end '
            'every name with the same word. Keep postal address and contact data out of the name. '
            'Do not copy Mediterranean Shipping or Misr Maritime.'
        )
        path = OUT / 'templates' / f'{sid}.json'
        atomic_write_json(path, template)
        catalog['templates'][sid]['template'] = pin(path)
    atomic_write_json(OUT / 'catalog.json', catalog)
    native = PaddleTemplateCatalog(ROOT, pin(OUT / 'catalog.json'))
    original = PaddleTemplateCatalog(ROOT, before_catalog)
    amendments, timings = [], Counter()
    for line in (DATA / 'synthetic-train.jsonl').open():
        row = json.loads(line)
        if row['documentId'] not in changes:
            continue
        change = changes[row['documentId']]
        candidate_path = ROOT / row['evidence']['candidate']['path']
        candidate = read(candidate_path)
        wording = read(row['evidence']['wording']['path'])
        t = time.perf_counter()
        old = original.render(candidate_path, wording_receipt=wording)
        timings['beforeSeconds'] += time.perf_counter() - t
        assert all(old[k] == row[k] for k in PUBLIC), 'baseline does not replay exactly'
        # Prompt-only template amendment must preserve the original replay exactly.
        assert native.render(candidate_path, wording_receipt=wording) == old
        assert wording['values']['delivery_name'] == change['before']
        wording['values']['delivery_name'] = change['after']
        wording['manualSuffixAmendment'] = {'before': change['before'], 'after': change['after'],
                                          'previousReceipt': row['evidence']['wording'], 'plan': pin(OUT / 'plan.json')}
        t = time.perf_counter()
        new = native.render(candidate_path, wording_receipt=wording)
        timings['afterSeconds'] += time.perf_counter() - t
        changed_lines = check_scoped_change(old, new, change['before'], change['after'])
        overlay = native._template(row['sourceDocumentId'])[0]['overlay']
        boundary = next(r for r in boundary_readback(new, candidate, overlay)
                        if r['path'] == '/parties/deliveryAgent')
        assert boundary['status'] == 'matched'
        assert any(o['kind'] == 'continuation_shared' for o in boundary['occurrences'])
        folder = OUT / 'records' / row['documentId']
        atomic_write_json(folder / 'before.json', row)
        atomic_write_json(folder / 'wording.json', wording)
        atomic_write_json(folder / 'rendered.json', new)
        review = {'status': 'PASS', 'scope': 'delivery-agent name ending only; original full review retained',
                  'priorEvidence': row['evidence'], 'plan': pin(OUT / 'plan.json'),
                  'changedLines': changed_lines, 'boundary': boundary,
                  'otherTargetsUnchanged': True, 'otherLinesAndCoordinatesUnchanged': True,
                  'beforeCatalog': before_catalog, 'afterCatalog': pin(OUT / 'catalog.json')}
        atomic_write_json(folder / 'review.json', review)
        updated = copy.deepcopy(row)
        updated.update({k: new[k] for k in PUBLIC})
        updated.update(text_view_digests(updated))
        updated['evidence'] = {'candidate': row['evidence']['candidate'],
                               'catalog': pin(OUT / 'catalog.json'), 'wording': pin(folder / 'wording.json'),
                               'rendered': pin(folder / 'rendered.json'), 'review': pin(folder / 'review.json'),
                               'previousRecord': pin(folder / 'before.json')}
        updated['agencyDiversityAmendment'] = pin(folder / 'review.json')
        atomic_write_json(folder / 'after.json', updated)
        amendments.append({'id': row['documentId'], 'before': pin(folder / 'before.json'),
                           'after': pin(folder / 'after.json'), 'review': pin(folder / 'review.json')})
        print(row['documentId'], change['after'], flush=True)
    assert len(amendments) == 40
    atomic_write_json(OUT / 'manifest-before.json', manifest)
    result = {'status': 'STAGED', 'changes': amendments, 'beforeManifest': pin(DATA / 'manifest.json'),
              'catalog': pin(OUT / 'catalog.json'), 'apiCalls': 0, 'timings': dict(timings),
              'elapsedSeconds': time.perf_counter() - started,
              'peakRssKiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    atomic_write_json(OUT / 'staged.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'changes'}))


def publish():
    staged = read(OUT / 'staged.json')
    assert staged['beforeManifest'] == pin(DATA / 'manifest.json')
    manifest = read(DATA / 'manifest.json')
    source = DATA / 'synthetic-train.jsonl'
    assert manifest['aggregateRecords']['synthetic-train'] == pin(source)
    changes = {r['id']: r for r in staged['changes']}
    count = 0
    with tempfile.NamedTemporaryFile(dir=DATA, prefix='.agency-diversity-', delete=False) as f:
        temporary = Path(f.name)
        for line in source.open('rb'):
            row = json.loads(line)
            if row['documentId'] in changes:
                change = changes[row['documentId']]
                assert row == read(change['before']['path'])
                assert change['after'] == pin(ROOT / change['after']['path'])
                line = (json.dumps(read(change['after']['path']), ensure_ascii=False, separators=(',', ':')) + '\n').encode()
                count += 1
            f.write(line)
        f.flush()
        os.fsync(f.fileno())
    assert count == 40
    receipt = {**staged, 'status': 'PASS', 'changedRecords': count, 'unchangedRecords': 10110,
               'realValidationAndNew750Unchanged': True, 'historicalEvidenceUnchanged': True,
               'before': manifest['aggregateRecords']['synthetic-train'],
               'after': {'path': str(source.relative_to(ROOT)), 'sha256': sha256_file(temporary)}}
    atomic_write_json(OUT / 'receipt.json', receipt)
    os.replace(temporary, source)
    manifest['aggregateRecords']['synthetic-train'] = receipt['after']
    manifest['agencyDiversityAmendment'] = pin(OUT / 'receipt.json')
    manifest['templateCatalog'] = staged['catalog']
    manifest['unchanged'] = [
        'Real/validation and the 750 new records remain unchanged. Historical batch certificates remain immutable. '
        'The active synthetic aggregate additionally applies textDigestAmendment and agencyDiversityAmendment; '
        'the latter changes 40 older delivery-agent names and their rendered suffix lines/geometry only.'
    ]
    atomic_write_json(DATA / 'manifest.json', manifest)
    print(json.dumps({'aggregate': pin(source), 'manifest': pin(DATA / 'manifest.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'stage', 'publish'])
    args = parser.parse_args()
    {'audit': audit, 'stage': stage, 'publish': publish}[args.action]()
