"""Certify route and care-of ownership edges exposed by fresh R31 draws.

Original catalogs and published samples stay immutable. The repaired authority
and native catalog are promoted only after the held Hong Kong case replays.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign
from document_ocr.synthesis.independent_geography import IndependentGeographyGate, load_independent_geography_certificates
from scripts.synthesis.rebase_reviewed_locality_frames import frame_proposals, rebase_owner_frames
from scripts.synthesis.rebase_reviewed_geography import validate_postal_preserving_rebase
from scripts.synthesis.extend_r31_boundaries import ROOT, OUT, CATALOG, read, pin, create_campaign, PaddleTemplateCatalog


def main():
    destination = OUT / 'locality-frame-repair'
    config = read(OUT/'batch-004/requested-config.json')
    sid = next(s for s in config['source_ids'] if s[4:12]=='fc68b5d9')
    before = _SourceOwnershipCampaign(ROOT, config).blueprint(sid)
    frames = frame_proposals(before)
    if [o['text'] for f in frames for o in f['occurrences']] != ['NHAVA SHEVA, INDIA','SOKHNA PORT, EGYPT']:
        raise ValueError('reviewed source route frames changed')
    owners = read(config['ownership'])
    owners['sources'][sid] = rebase_owner_frames(owners['sources'][sid],frames)
    atomic_write_json(destination/'ownership.json',owners)
    config['ownership'] = str((destination/'ownership.json').relative_to(ROOT))
    after = _SourceOwnershipCampaign(ROOT,config).blueprint(sid)
    if before.source!=after.source or before.target!=after.target:
        raise ValueError('frame repair changed source or targets')
    old_keys={k for f in frames for k in f['oldKeys']}
    new_keys={f['key'] for f in frames}
    if [r for r in before.regions if r.key not in old_keys] != [r for r in after.regions if r.key not in new_keys]:
        raise ValueError('unrelated ownership changed')
    reference=config['independent']['geography_certificates']
    gate=IndependentGeographyGate(load_independent_geography_certificates((ROOT/reference['path']).read_bytes(),expected_sha256=reference['sha256']))
    certificates=read(reference['path'])
    if sid in gate.certificates:
        pins=validate_postal_preserving_rebase(gate,before,after)
        certificates['sources'][sid].update(effective_contract_sha256=pins[0],ownership_sha256=pins[1])
    else:
        gate._certificate(before,allow_prepared=False);gate._certificate(after,allow_prepared=False)
    atomic_write_json(destination/'geography_certificates.json',certificates)
    config['independent']['geography_certificates']=pin(destination/'geography_certificates.json')
    catalog=read(CATALOG);template=read(catalog['templates'][sid]['template']['path'])
    source=read(template['source']['path'])
    # Exact reviewed Paddle lines own each complete pair and its punctuation.
    for frame,region,offset in zip(frames,[49,58],[739,907],strict=True):
        text=source['pages'][0]['regions'][region]['text']
        if text!=frame['occurrences'][0]['text']:
            raise ValueError('native frame no longer matches reviewed text')
        template['overlay']['overrides'].append({'key':frame['key'],'byteStart':offset,'spans':[[0,region,0,len(text)]]})
    atomic_write_json(destination/f'{sid}.json',template)
    catalog['templates'][sid]['template']=pin(destination/f'{sid}.json')
    atomic_write_json(destination/'catalog.json',catalog)
    prepared=read(read(OUT/'batch-004/preparation.json')['preparedConfig']['path'])
    prepared['ownership']=config['ownership']
    prepared['independent']['geography_certificates']=config['independent']['geography_certificates']
    campaign=create_campaign(ROOT,prepared)
    candidate=campaign.render(sid,1,persist=False)
    candidate_path=destination/'candidate.json';atomic_write_json(candidate_path,candidate)
    native=PaddleTemplateCatalog(ROOT,pin(destination/'catalog.json'))
    spec=native.wording_specification(candidate_path)
    if set(spec['requested'])!={'signature'}:
        raise ValueError('reviewed additional owner set changed')
    wording={k:spec[k] for k in ['documentId','sourceFingerprint','candidateSha256']}
    wording.update(status='REVIEWED',values={'signature':'Amit Verma'},reviewReason='Main-agent authored fictional signatory; exact requested short-name owner only.')
    atomic_write_json(destination/'wording.json',wording)
    rendered=native.render(candidate_path,wording_receipt=wording)
    loading=rendered['target']['route']['portOfLoading']
    if loading!={'name':'HONG KONG'}:
        raise ValueError(f'homonym projection differs: {loading}')
    if 'HONG KONG, HONG KONG' in rendered['joinedRawText'].upper():
        raise ValueError('repeated homonym survived')
    atomic_write_json(destination/'rendered.json',rendered)
    # Also replay an existing, non-homonymous fresh candidate unchanged.
    control=next(p for p in (OUT/'batch-005/sampling/candidates').glob('*.json') if read(p)['sourceDocumentId']==sid)
    control_wording=read(OUT/'batch-005/wording-reviewed'/control.name)
    old_native=PaddleTemplateCatalog(ROOT,pin(CATALOG)).render(control,wording_receipt=control_wording)
    new_native=native.render(control,wording_receipt=control_wording)
    if old_native!=new_native:
        raise ValueError('existing split-owner replay changed')
    care_sid=next(s for s in catalog['templates'] if s[4:12]=='d684b168')
    care_template=read(catalog['templates'][care_sid]['template']['path'])
    care_source=read(care_template['source']['path'])
    if care_source['pages'][0]['regions'][4]['text']!='MIAMI GLOBAL LINES C/O':
        raise ValueError('reviewed static care-of caption changed')
    # C/O belongs to the address but shares the preceding name's physical line.
    # Include that actual occurrence, not unrelated document-wide O/o tokens.
    care_template['overlay'].setdefault('sourceCaseRegions',{})['/parties/shipper/addressLine']=[[0,4],[0,6],[0,7],[0,9]]
    atomic_write_json(destination/f'{care_sid}.json',care_template)
    catalog['templates'][care_sid]['template']=pin(destination/f'{care_sid}.json')
    atomic_write_json(destination/'catalog.json',catalog)
    native=PaddleTemplateCatalog(ROOT,pin(destination/'catalog.json'))
    care_candidate=next((OUT/'batch-007/sampling/candidates').glob('*65c77354dc.json'))
    care_rendered=native.render(care_candidate)
    if care_rendered['target']['parties']['shipper']['addressLine']!='C/O Te Maiya Trading Stores Betio Causeway Road Tarawa':
        raise ValueError('care-of occurrence projection differs')
    atomic_write_json(destination/'care-of-rendered.json',care_rendered)
    care_controls=[];timings={'previous':0.0,'repaired':0.0}
    previous=PaddleTemplateCatalog(ROOT,pin(CATALOG))
    from scripts.synthesis.extend_r31_boundaries import DATA
    for row in map(json.loads,(DATA/'synthetic-train.jsonl').open()):
        if row['sourceDocumentId']!=care_sid:continue
        evidence=row['evidence']; path=ROOT/evidence['candidate']['path']
        wording=read(evidence['wording']['path']) if evidence.get('wording') else None
        start=time.perf_counter();old=previous.render(path,wording_receipt=wording);timings['previous']+=time.perf_counter()-start
        start=time.perf_counter();new=native.render(path,wording_receipt=wording);timings['repaired']+=time.perf_counter()-start
        for field in ['pages','target','positionedText','joinedRawText','bboxText']:
            if old[field]!=new[field]:raise ValueError('care-of scope changed an existing public record')
        care_controls.append(row['documentId'])
    production=read('configs/synthesis/mpci_bl_curated_v7_paddle.json')
    production['ownership']=config['ownership']
    production['independent']['geography_certificates']=config['independent']['geography_certificates']
    atomic_write_json(destination/'production-config.json',production)
    atomic_write_json(destination/'review.json',{'status':'PASS','frames':frames,'sourceAndTargetsUnchanged':True,
        'unrelatedOwnersUnchanged':True,'heldHongKongReplay':pin(destination/'rendered.json'),
        'existingSplitOwnerReplayUnchanged':pin(control),'nativeCatalog':pin(destination/'catalog.json'),
        'careOfHeldReplay':pin(destination/'care-of-rendered.json'),'careOfExistingControls':care_controls,'careOfControlSeconds':timings,
        'apiCalls':0,'promotionConfig':pin(destination/'production-config.json')})
    print(json.dumps({'status':'PASS','repairedFrames':len(frames),'apiCalls':0}))


if __name__=='__main__':main()
