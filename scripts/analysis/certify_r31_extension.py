"""Read back the published extension, replay it, and measure actual coverage."""
from __future__ import annotations

import argparse
import json
import re
import resource
import time
from collections import Counter
from decimal import Decimal

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.labeling_agents.source_casing import selected_fields
from document_ocr.spatial_inputs.paddle import pages_from_record,serialize,text_view_digests
from scripts.synthesis.extend_r31_boundaries import ROOT,OUT,DATA,REPORT,read,pin,PaddleTemplateCatalog


def parties(target):
    for role,value in target.get('parties',{}).items():
        for item in value if isinstance(value,list) else [value]:
            if isinstance(item,dict):yield role,item


def certify(replay=True):
    started=time.perf_counter();manifest=read(DATA/'manifest.json');baseline=read(OUT/'baseline/manifest.json')
    for split in ['real-train','validation']:
        assert manifest['aggregateRecords'][split]==baseline['aggregateRecords'][split]
        assert sha256_file(ROOT/baseline['aggregateRecords'][split]['path'])==baseline['aggregateRecords'][split]['sha256']
    amendments={};suffix_changes={};description_changes={}
    digest_aggregate=manifest['aggregateRecords']['synthetic-train']
    if manifest.get('readinessDescriptionAmendment'):
        ref=manifest['readinessDescriptionAmendment'];assert sha256_file(ROOT/ref['path'])==ref['sha256']
        description_receipt=read(ref['path'])
        assert description_receipt['after']==digest_aggregate
        digest_aggregate=description_receipt['before']
        for change in description_receipt['changes']:
            for key in ['before','after','review']:
                ref=change[key];assert sha256_file(ROOT/ref['path'])==ref['sha256']
            description_changes[change['id']]=change
    if manifest.get('agencyDiversityAmendment'):
        ref=manifest['agencyDiversityAmendment'];assert sha256_file(ROOT/ref['path'])==ref['sha256']
        suffix_receipt=read(ref['path'])
        assert suffix_receipt['after']==digest_aggregate
        digest_aggregate=suffix_receipt['before']
        for change in suffix_receipt['changes']:
            for key in ['before','after','review']:
                ref=change[key];assert sha256_file(ROOT/ref['path'])==ref['sha256']
            suffix_changes[change['id']]=change
    if manifest.get('textDigestAmendment'):
        ref=manifest['textDigestAmendment'];assert sha256_file(ROOT/ref['path'])==ref['sha256']
        receipt=read(ref['path']);ref=receipt['changes']
        assert sha256_file(ROOT/ref['path'])==ref['sha256']
        assert receipt['after']==digest_aggregate
        amendments={r['id']:r['fields'] for r in read(ref['path'])}
    changed=set();suffix_seen=set();description_seen=set()
    current=(DATA/'synthetic-train.jsonl').open('rb')
    with (OUT/'baseline/synthetic-train.jsonl').open('rb') as original:
        for old in original:
            new=current.readline()
            if old!=new:
                before,after=json.loads(old),json.loads(new)
                if before['documentId'] in description_changes:
                    change=description_changes[before['documentId']]
                    assert after==read(change['after']['path'])
                    after=read(change['before']['path'])
                    description_seen.add(before['documentId'])
                if before['documentId'] in suffix_changes:
                    change=suffix_changes[before['documentId']]
                    assert after==read(change['after']['path'])
                    after=read(change['before']['path'])
                    suffix_seen.add(before['documentId'])
                fields=amendments[before['documentId']]
                expected=text_view_digests(after)
                for key,delta in fields.items():
                    assert before[key]==delta['before']
                    assert after[key]==delta['after']==expected[key]
                    before[key]=after[key]
                assert before==after
                changed.add(after['documentId'])
    assert changed==set(amendments)
    assert suffix_seen==set(suffix_changes)
    assert description_seen==set(description_changes)
    new=[json.loads(line) for line in current];current.close()
    assert len(new)==750
    ids=set();validation={r['documentId'] for r in map(json.loads,(DATA/'validation.jsonl').open())}
    for split in ['real-train','synthetic-train','validation']:
        assert sha256_file(ROOT/manifest['aggregateRecords'][split]['path'])==manifest['aggregateRecords'][split]['sha256']
        for row in map(json.loads,(DATA/f'{split}.jsonl').open()):
            assert row['documentId'] not in ids;ids.add(row['documentId'])
            if split=='synthetic-train':
                assert all(row[k]==v for k,v in text_view_digests(row).items())
    assert len(ids)==10810
    catalogs={};counts=Counter();parents=Counter();packages=Counter();hs=set();names=Counter();roles=Counter();styles=Counter()
    active=PaddleTemplateCatalog(ROOT,manifest['templateCatalog']) if replay else None
    boundary=Counter();boundary_parents=Counter();examples=[]
    audit_rows={r['id']:r for f in OUT.glob('batch-*/audit.json') for r in read(f)['rows']}
    reviews={r['id']:r for f in OUT.glob('batch-*/native-review.json') for r in read(f)['results']}
    for row in new:
        assert row['sourceDocumentId'] not in validation
        assert BillOfLadingExtractionV7DirectTarget.model_validate_json(json.dumps(row['target'])).canonical_target()==row['target']
        p=pages_from_record(row)
        for mode,field in [('none','joinedRawText'),('centroid','positionedText'),('bbox','bboxText')]:assert serialize(p,geometry=mode)==row[field]
        assert all('\n' not in value and '\r' not in value for _,value in selected_fields(row['target']))
        assert row['sourceCasing']['policy']=='printed-party-and-goods-v1'
        assert dict(selected_fields(row['target']))=={f['path']:f['after'] for f in row['sourceCasing']['fields']}
        e=row['evidence']
        for key in ['candidate','rendered','wording','review','catalog']:
            if e.get(key):assert sha256_file(ROOT/e[key]['path'])==e[key]['sha256']
        if replay:
            ref=e['catalog']
            if ref['path'] not in catalogs:catalogs[ref['path']]=PaddleTemplateCatalog(ROOT,ref)
            cat=catalogs[ref['path']]
            replayed=cat.render(ROOT/e['candidate']['path'],wording_receipt=read(e['wording']['path']) if e.get('wording') else None)
            for field in ['pages','target','positionedText','joinedRawText','bboxText','sourceCasing']:assert row[field]==replayed[field]
            active_render=active.render(ROOT/e['candidate']['path'],wording_receipt=read(e['wording']['path']) if e.get('wording') else None)
            for field in ['pages','target','positionedText','joinedRawText','bboxText']:assert row[field]==active_render[field]
        counts['documents']+=1;counts['lines']+=sum(len(page.regions) for page in p)
        t=row['target'];cs=t.get('containerInformation',[]);gs=t.get('goodsItemDetails',[]);ps=[x for g in gs for x in g.get('numberAndTypeOfPackages',[])]
        counts['multiContainer']+=len(cs)>1;counts['mixedEquipment']+=len({(c.get('typeCategory'),c.get('sizeCategory')) for c in cs})>1
        counts['multiPackageRow']+=len(ps)>1;counts['thermal']+=any(c.get('temperatureSetpoint') for c in cs)
        counts['dg']+=any(g.get('dangerousGoods') for g in gs);counts['multiDg']+=any(len(g.get('dangerousGoods',[]))>1 for g in gs)
        parents[row['sourceDocumentId'][4:12]]+=1;packages.update(x['typeCategory'] for x in ps)
        hs.update(re.sub(r'\D','',str(h))[:6] for g in gs for h in g.get('hsCodes',[]))
        for role,party in parties(t):
            roles[role]+=1
            if 'name' in party:names[party['name'].casefold()]+=1
        for field,value in selected_fields(t):styles['uppercase' if value.isupper() else 'lowercase' if value.islower() else 'mixed_or_title']+=1
        a=audit_rows[row['documentId']];boundary[a['boundary']]+=1;boundary['continuationRoles']+=a['continuationRoles']
        if a['boundary']!='clear_or_unresolved':boundary_parents[row['sourceDocumentId'][4:12]]+=1
        r=reviews[row['documentId']]
        for b in r['boundaries']:
            kinds={o['kind'] for o in b['occurrences']}
            if 'continuation_shared' not in kinds and 'whole_name_shared' in kinds:boundary['wholeNameRoles']+=1
            if len(examples)<12 and any(o['kind']=='continuation_shared' for o in b['occurrences']):
                if not any(x['parent']==row['sourceDocumentId'][4:12] for x in examples):examples.append({'id':row['documentId'],'parent':row['sourceDocumentId'][4:12],'role':b['path'],'name':b['name'],'address':b['address'],'line':next(o['boundary']['text'] for o in b['occurrences'] if o['kind']=='continuation_shared')})
    batches=[json.loads(p.read_text()) for p in sorted((DATA/'batches').glob('boundary-extension-*.certificate.json'))]
    assert [b['count'] for b in batches]==[100]*7+[50]
    cost=Decimal(0);calls=Counter();tokens=Counter()
    for path in OUT.glob('batch-*/**/calls/*.json'):
        d=read(path)
        if 'costUsd' not in d:continue
        cost+=Decimal(str(d['costUsd']));calls['failed' if d.get('error') else 'successful']+=1
        for k,v in d.get('usage',{}).items():
            if isinstance(v,int):tokens[k]+=v
    report={'status':'PASS','new':dict(counts),'parents':dict(parents),'parentCount':len(parents),'boundary':dict(boundary),'difficultParents':len(boundary_parents),
        'packageRowsByCategory':dict(packages),'packageCategoryCount':len(packages),'hsSixDigitCount':len(hs),'roles':dict(roles),'fieldCasing':dict(styles),
        'uniquePartyNames':len(names),'partyNameOccurrences':sum(names.values()),'mostRepeatedNames':names.most_common(12),'agencyNameOccurrences':sum(v for k,v in names.items() if 'agency' in k),
        'batchSizes':[b['count'] for b in batches],'original10000PayloadAndValidation60Unchanged':not (suffix_changes or description_changes),
        'legacySuffixAmendedRecords':len(suffix_changes),'allOtherOriginalPayloadAndValidation60Unchanged':True,
        'legacyDescriptionAmendedRecords':len(description_changes),
        'legacyTextMetadataAmendedRecords':len(amendments),'exactReplay':replay,'activeCatalogPublicReplay':replay,'zeroDuplicateIdsAndValidationParents':True,
        'measuredCoordinateCoverage':1.0,'examples':examples,'callReceipts':dict(calls),'allAttemptCostUsd':str(cost),'usageFields':dict(tokens),
        'seconds':time.perf_counter()-started,'peakRssKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'manifest':pin(DATA/'manifest.json')}
    atomic_write_json(REPORT/'final-certification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['examples','parents','packageRowsByCategory','mostRepeatedNames']},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--skip-replay',action='store_true');a=p.parse_args();certify(not a.skip_replay)
