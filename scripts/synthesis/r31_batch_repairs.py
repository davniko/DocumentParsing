"""Explicit, reviewed first-batch repairs; never an automatic acceptance gate."""
import asyncio
import copy

from scripts.synthesis.extend_r31_boundaries import *
from document_ocr.synthesis.curated_contacts import request_hash as contact_request_hash, contact_output_type, unpack_contacts


async def repair():
    directory = OUT / 'batch-001'
    config = read(read(directory/'preparation.json')['preparedConfig']['path'])
    campaign = create_campaign(ROOT, config)
    sid = next(s for s in campaign.source_ids if s[4:12] == '64c42723')
    parties = campaign.contact_requests(sid)
    expected = ['GULF EAST FORWARDING LLC', 'FUJAIRAH LINE AGENCIES LLC', 'RAK PORT CLEARING & FORWARDING LLC']
    assert [p.name for p in parties] == expected
    path = campaign.output/'contacts'/f'{sid}.json'
    if not path.exists():
        output = contact_output_type(parties).model_validate({
            'p0': {'c0': 'ge201-forwarding@gulfeastlogistics.ae'},
            'p1': {'c0': 'fl202-agency@fujairahline.ae'},
            'p2': {'c0': 'rk203-clearance@rakportfreight.ae'},
        })
        atomic_write_json(path, {'requestSha256':contact_request_hash(parties),
            'output':output.model_dump(mode='json'), 'values':unpack_contacts(output,parties),
            'review':'Main-agent fictional contact correction after two invalid provider responses.'})
    await campaign.generate_contacts(sid)
    corrections = {
        'd25918fa3b': ('TOWER 6, PALM GATE OFFICES\nTOWER 6, PALM GATE OFFICES',
                       'TOWER 6, PALM GATE OFFICES\nRECEIVING SUITE 14'),
        '757a8dc1be': ('PLOT 88, SECTOR SH-3\nPLOT 88, SECTOR SH-3',
                       'PLOT 88, SECTOR SH-3\nWAREHOUSE 12, LOADING BAY 4'),
    }
    for suffix, (before, after) in corrections.items():
        path = next((campaign.output/'wording').glob(f'*{suffix}.json'))
        row = read(path)
        if before in row['values']['postal_2']:
            atomic_write_json(directory/'history/wording'/path.name,row)
            row['values']['postal_2']=row['values']['postal_2'].replace(before,after)
            row['mainAgentCorrection']={'field':'postal_2','before':before,'after':after,
                                         'reason':'Replace duplicate premise line; preserve requested two address lines.'}
            atomic_write_json(path,row)
    changes=[]
    for sid in campaign.source_ids:
        if sid[4:12] not in {'64c42723','c68c55e0','2d5864aa','f1f044b2'}:continue
        if sid[4:12] == 'f1f044b2':
            await campaign.generate(sid)
        for variant in range(1,campaign.variant_counts[sid]+1):
            print('repair-render',sid[4:12],variant,flush=True)
            fresh=campaign.render(sid,variant,persist=False)
            path=campaign.output/'candidates'/f"{fresh['documentId']}.json"
            if path.exists() and read(path)==fresh:continue
            if path.exists():
                atomic_write_json(directory/'history/candidates'/path.name,read(path))
            atomic_write_json(path,fresh)
            # Existing Paddle-only wording retains the same goods/role/locality
            # context for these scoped edits. Pin the explicit reviewed rebase.
            for folder in ('wording-proposals','wording-reviewed'):
                wp=directory/folder/path.name
                if wp.exists():
                    value=read(wp)
                    atomic_write_json(directory/'history'/folder/path.name,value)
                    value['reviewedCandidateRebase']={'beforeSha256':value['candidateSha256'],
                        'reason':'Reviewed premise correction or deterministic clause wrapper repair; native extra wording unchanged.'}
                    value['candidateSha256']=sha256_file(path)
                    atomic_write_json(wp,value)
            changes.append(fresh['documentId'])
    atomic_write_json(directory/'scoped-repairs.json', {'changed':changes,
        'preparationReplayNote':'Recovered earlier package proposals changed adaptive reservation counts, hence the later c68c55e0:2 package draw; its full new numeric/wording authority is rendered together. Previous unpublished candidate archived.'})
    print(changes)


if __name__=='__main__':asyncio.run(repair())
