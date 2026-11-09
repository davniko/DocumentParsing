"""Publish only hash-pinned, personally reviewed R31 extension batches."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from document_ocr.atomic import atomic_write_json, atomic_publish_bytes
from document_ocr.hashing import sha256_file
from document_ocr.spatial_inputs.paddle import text_view_digests
from scripts.synthesis.extend_r31_boundaries import DATA, OUT, ROOT, CATALOG, read, pin


def verify(reference):
    if sha256_file(ROOT/reference['path']) != reference['sha256']:
        raise ValueError(f"reviewed artifact changed: {reference['path']}")


def publish(count):
    if count not in (100,50):raise ValueError('publication batches are 100, or the final 50')
    manifest=read(DATA/'manifest.json')
    for reference in manifest['aggregateRecords'].values():verify(reference)
    baseline=OUT/'baseline'
    baseline.mkdir(parents=True,exist_ok=True)
    if not (baseline/'manifest.json').exists():
        if manifest['stage']!='A-casing-only':raise ValueError('missing Stage A baseline')
        for split,reference in manifest['aggregateRecords'].items():
            verify(reference)
        shutil.copy2(DATA/'synthetic-train.jsonl',baseline/'synthetic-train.jsonl')
        atomic_write_json(baseline/'manifest.json',manifest)
    original=read(baseline/'manifest.json')
    for split in ('real-train','validation'):
        verify(original['aggregateRecords'][split])
    existing=set();validation_ids=set()
    for split in ('real-train','synthetic-train','validation'):
        for row in map(json.loads,(DATA/f'{split}.jsonl').open()):
            existing.add(row['documentId'])
            if split=='validation':validation_ids.add(row['documentId'])
    added=len(existing)-10060
    if added+count>750:raise ValueError('approved extension is 750 records')
    choices=[]
    for path in sorted(OUT.glob('batch-*/acceptance.json')):
        decision=read(path)
        if decision['status']!='MAIN_AGENT_REVIEWED':raise ValueError('unreviewed decision')
        verify(decision['nativeReview']);verify(decision['audit'])
        native=read(decision['nativeReview']['path'])
        verify(native['catalog'])
        if native['catalog']!=pin(CATALOG):raise ValueError('batch uses another catalog')
        lookup={r['id']:r for r in native['results']}
        for identity in decision['accepted']:
            if identity in existing:continue
            if identity not in lookup:raise ValueError('acceptance names absent native record')
            evidence=lookup[identity]
            verify(evidence['candidate']);verify(evidence['rendered'])
            if evidence['wording']:verify(evidence['wording'])
            choices.append((evidence,pin(path)))
    if len(choices)<count:raise ValueError(f'only {len(choices)} reviewed unpublished records; need {count}')
    choices=choices[:count]
    if len({r['id'] for r,_ in choices})!=count:raise ValueError('duplicate new ID')
    rows=[]
    for evidence,review in choices:
        rendered=read(evidence['rendered']['path'])
        if rendered['sourceDocumentId'] in validation_ids:
            raise ValueError('a validation identity cannot parent a training descendant')
        row={k:rendered[k] for k in ('documentId','sourceDocumentId','target','pages','joinedRawText','positionedText','bboxText')}
        row.update(text_view_digests(row))
        row['evidence']={k:evidence[k] for k in ('candidate','rendered','wording')}
        row['evidence'].update(catalog=pin(CATALOG),review=review)
        row['sourceCasing']=rendered['sourceCasing']
        rows.append(row)
    index=added//100+1
    batch=DATA/'batches'/f'boundary-extension-{index:03d}.jsonl'
    payload=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows).encode()
    atomic_publish_bytes(batch,payload)
    certificate=batch.with_suffix('.certificate.json')
    atomic_write_json(certificate,{'count':count,'records':pin(batch),'catalog':pin(CATALOG),
        'reviews':list({r['path']:r for _,r in choices}.values()),
        'preservedStageA':pin(baseline/'manifest.json'),
        'status':'ACCEPTED','scope':'New donor-free Paddle records; case-preserved targets; personally reviewed rendered boundaries and cargo controls.'})
    aggregate=DATA/'synthetic-train.jsonl'
    with tempfile.NamedTemporaryFile(dir=DATA,prefix='.extension-',delete=False) as handle:
        temporary=Path(handle.name)
        with aggregate.open('rb') as src:shutil.copyfileobj(src,handle,1024*1024)
        handle.write(payload);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,aggregate)
    manifest['aggregateRecords']['synthetic-train']=pin(aggregate)
    manifest['batches'].append({'count':count,'records':pin(batch),'certificate':pin(certificate)})
    manifest['stage']='A-casing-plus-B-boundary-extension'
    manifest['stageB']={'status':'complete' if added+count==750 else 'in-progress',
        'addedSynthetic':added+count,'target':750,'trainTotal':10000+added+count,
        'stageABaseline':pin(baseline/'manifest.json')}
    manifest['templateCatalog']=pin(CATALOG)
    manifest['unchanged']=['The original 10000 training records and 60 validation records remain byte-identical to Stage A.']
    atomic_write_json(DATA/'manifest.json',manifest)
    print(json.dumps({'published':count,'added':added+count,'trainTotal':10000+added+count,'batch':str(batch)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--count',type=int,default=100)
    publish(parser.parse_args().count)
