"""Deterministic extension checks and compact main-agent review packets."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from scripts.synthesis.extend_r31_boundaries import ROOT, OUT, read, pin
from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_bytes
from document_ocr.spatial_inputs.paddle import pages_from_record, serialize
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.labeling_agents.source_casing import selected_fields


def audit(batch):
    directory=OUT/f'batch-{batch:03d}'
    review=read(directory/'native-review.json')
    rows=[]; counts=Counter(); categories=Counter(); flags=[]
    for result in review['results']:
        record=read(result['rendered']['path']); target=record['target']
        assert BillOfLadingExtractionV7DirectTarget.model_validate_json(json.dumps(target)).canonical_target()==target
        pages=pages_from_record(record)
        for mode,field in [('none','joinedRawText'),('centroid','positionedText'),('bbox','bboxText')]:
            assert serialize(pages,geometry=mode)==record[field]
        assert record['sourceCasing']['policy']=='printed-party-and-goods-v1'
        assert {f['path']:f['after'] for f in record['sourceCasing']['fields']}==dict(selected_fields(target))
        findings=[]
        for field,value in selected_fields(target):
            if '\n' in value or '\r' in value:findings.append({'field':field,'kind':'target_newline'})
            words=value.split()
            for length in range(3,min(len(words)//2,12)+1):
                if any(words[i:i+length]==words[i+length:i+2*length] for i in range(len(words)-2*length+1)):
                    findings.append({'field':field,'kind':'repeated_phrase','value':value});break
            if re.search(r'(?i)fictional|generate|do not|your output|placeholder',value):
                findings.append({'field':field,'kind':'instruction_screen','value':value})
        role_kinds=[]
        for b in result['boundaries']:
            kinds={o['kind'] for o in b['occurrences']}
            role_kinds.append('continuation' if 'continuation_shared' in kinds else 'whole' if 'whole_name_shared' in kinds else 'separate' if kinds else 'unresolved')
        groups=target.get('goodsItemDetails',[])
        packages=[p for g in groups for p in g.get('numberAndTypeOfPackages',[])]
        cs=target.get('containerInformation',[])
        kind='continuation' if 'continuation' in role_kinds else 'whole' if 'whole' in role_kinds else 'clear_or_unresolved'
        counts[kind]+=1;counts['continuationRoles']+=role_kinds.count('continuation')
        counts['wholeRoles']+=role_kinds.count('whole');counts['unresolvedRoles']+=role_kinds.count('unresolved')
        counts['multiContainer']+=len(cs)>1
        counts['multiPackageRow']+=len(packages)>1
        counts['dg']+=any(g.get('dangerousGoods') for g in groups)
        counts['thermal']+=any(c.get('temperatureSetpoint') for c in cs)
        counts['forwarder']+=bool(target.get('parties',{}).get('forwardingAgent'))
        types={(c.get('typeCategory'),c.get('sizeCategory')) for c in cs}
        counts['mixedEquipment']+=len(types)>1
        categories.update(p['typeCategory'] for p in packages)
        rows.append({'id':record['documentId'],'parent':record['sourceDocumentId'], 'boundary':kind,
            'continuationRoles':role_kinds.count('continuation'),'findings':findings,
            'goods':[g.get('description','') for g in groups],
            'packages':packages,'containers':len(cs), 'dg':any(g.get('dangerousGoods') for g in groups)})
        if findings:flags.append(rows[-1])
    receipt={'nativeReview':pin(directory/'native-review.json'),'count':len(rows),'counts':dict(counts),
             'categories':dict(categories),'flags':flags,'rows':rows}
    atomic_write_json(directory/'audit.json',receipt)
    print({'count':len(rows),'counts':dict(counts),'flagged':len(flags)})
    for i,r in enumerate(rows):
        print(i,r['id'][-10:],r['parent'][4:12],r['boundary'],r['containers'],
              '/'.join(p['typeCategory'].removeprefix('PACKAGE_') for p in r['packages']),
              ' | '.join(r['goods'])[:260])
    for f in flags:print('FLAG',f['id'],f['findings'])
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True)
    audit(p.parse_args().batch)
