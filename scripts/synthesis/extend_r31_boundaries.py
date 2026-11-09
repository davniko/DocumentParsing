"""Staged, resumable R31 boundary-extension campaign and evidence readback."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import time
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.spatial_inputs.paddle import pages_from_record
from document_ocr.synthesis.paddle_render import spans_for
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog
from document_ocr.synthesis.curated_campaign import create_campaign, wording_request_hash
from document_ocr.synthesis.independent_preparation import inventory_sources, prepare_campaign
from document_ocr.synthesis.paddle_wording import propose
from document_ocr.synthesis.curated import CuratedSynthesisConfig, Runner
from document_ocr.synthesis.curated_wording import wording_output_type
from scripts.analysis.r31_boundary_quality import boundary_readback

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/curated/mpci-bl-r31-paddle-centroids-v1"
OUT = ROOT / "artifacts/synthesis-campaigns/r31-boundaries-20261109"
REPORT = ROOT / "docs/analysis/r31-boundary-extension-20261109"
BASE_CATALOG = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-paddle-reviewed/r31/printed-casing-20261109/catalog.json"
CATALOG = OUT / "catalog/catalog.json"


def read(path):
    return json.loads((ROOT / path).read_text())


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def matching_override(overlay, edit):
    matches = [o for o in overlay["overrides"] if o["key"] == edit["key"]
               and o.get("byteStart", edit["byteStart"]) == edit["byteStart"]
               and o.get("byteEnd", edit["byteEnd"]) == edit["byteEnd"]]
    if len(matches) > 1:
        raise ValueError("ambiguous owner")
    return matches[0] if matches else None


def inventory():
    """Propose only existing shared-line name frames; never invent an adjacency."""
    plan = read("docs/analysis/r31-casing-migration-20261109/boundary-plan.json")
    selected = set(plan["real-train"]["admittedSharedParents"]) | set(plan["synthetic-train"]["admittedSharedParents"])
    retained = {}
    for row in map(json.loads, (DATA / "synthetic-train.jsonl").open()):
        if row["sourceDocumentId"] in selected:
            retained.setdefault(row["sourceDocumentId"], row)
    cat = PaddleTemplateCatalog(ROOT, pin(BASE_CATALOG))
    result = []
    for sid in sorted(selected):
        template, source, transfer, _ = cat._template(sid)
        pages = pages_from_record(source)
        row = retained[sid]
        candidate = read(row["evidence"]["candidate"]["path"])
        overlay = copy.deepcopy(template["overlay"])
        lookup = {s["owner"]: s for s in transfer["slots"]}
        names, addresses = [], []
        for edit in candidate["proof"]["edits"]:
            paths = edit.get("targetPaths", [])
            kind = "name" if any(p.startswith("documentPatch.parties.") and p.endswith(".name") for p in paths) else "address" if any(p.endswith(".addressLine") for p in paths) else None
            if not kind:
                continue
            rule = matching_override(overlay, edit)
            if rule and rule.get("omit"):
                continue
            owner = f"{edit['key']}@{edit['byteStart']}:{edit['byteEnd']}"
            if rule and any(k in rule for k in ("fragments", "regions", "spans")):
                spans = [vars(s) for s in spans_for(rule, pages)]
            else:
                slot = lookup.get(owner)
                if slot is None or slot["status"] != "exact_measured_owner_candidate":
                    continue
                spans = slot["spans"]
            entry = {"edit": edit, "spans": spans, "rule": rule}
            (names if kind == "name" else addresses).append(entry)
        amendments, boundaries = [], []
        for name in names:
            ns = name["spans"]
            for address in addresses:
                ads = address["spans"]
                if not any(n["page"] == a["page"] and n["region"] == a["region"] and n["end"] <= a["start"] for n in ns for a in ads):
                    continue
                boundaries.append({"name": name["edit"]["after"], "address": address["edit"]["after"],
                                   "nameSpans": ns, "addressSpans": ads, "paths": name["edit"]["targetPaths"]})
                refs = {(s["page"], s["region"]) for s in ns}
                if len(ns) > 1 and len(refs) == len(ns):
                    rule = name["rule"]
                    if rule is None:
                        rule = {"key": name["edit"]["key"], "byteStart": name["edit"]["byteStart"],
                                "byteEnd": name["edit"]["byteEnd"],
                                "spans": [[s[k] for k in ("page", "region", "start", "end")] for s in ns]}
                        overlay["overrides"].append(rule)
                        name["rule"] = rule
                    if not rule.get("separateRegionFrames"):
                        rule["separateRegionFrames"] = True
                        amendments.append(copy.deepcopy(rule))
        result.append({"source": sid, "boundaries": boundaries, "amendments": amendments,
                       "overlay": overlay, "control": row["evidence"]})
    atomic_write_json(REPORT / "inventory.json", result)
    print(json.dumps({"templates": len(result), "shared": sum(bool(r['boundaries']) for r in result),
                      "amended": sum(bool(r['amendments']) for r in result)}))


def controls():
    manifest = read(BASE_CATALOG)
    results = []
    inventory = read(REPORT / "inventory.json")
    for item in inventory:
        if not item["amendments"]:
            continue
        sid = item["source"]
        template = read(manifest["templates"][sid]["template"]["path"])
        template["overlay"] = item["overlay"]
        path = CATALOG.parent / "templates" / f"{sid}.json"
        atomic_write_json(path, template)
        manifest["templates"][sid]["template"] = pin(path)
    manifest["boundaryParentCatalog"] = pin(BASE_CATALOG)
    atomic_write_json(CATALOG, manifest)
    cat = PaddleTemplateCatalog(ROOT, pin(CATALOG))
    base = PaddleTemplateCatalog(ROOT, pin(BASE_CATALOG))
    for item in inventory:
        evidence = item["control"]
        path = ROOT / evidence["candidate"]["path"]
        wording = read(evidence["wording"]["path"]) if evidence.get("wording") else None
        pair = []
        for catalog in (base, cat):
            started = time.perf_counter()
            try:
                rendered = catalog.render(path, wording_receipt=wording)
                template,_,_,candidate = catalog.shipment(path)
                pair.append({"status": "PASS", "seconds": time.perf_counter()-started,
                             "parties": rendered['target'].get('parties'),
                             "text": rendered["joinedRawText"], "target": rendered["target"],
                             "boundaries":boundary_readback(rendered,candidate,template['overlay'])})
            except Exception as exc:
                pair.append({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        results.append({"source": item["source"], "amended": bool(item['amendments']),
                        "baseline": pair[0], "proposed": pair[1]})
    atomic_write_json(REPORT / "retained-controls.json", results)
    for r in results:
        print(r['source'][4:12], r['amended'], r['baseline']['status'], r['proposed']['status'], r['proposed'].get('error',''))


def stage(batch):
    config = read("configs/synthesis/mpci_bl_curated_v7_paddle.json")
    inventory = read(REPORT / "inventory.json")
    # 2418b6d9 does not establish complete canonical equipment rows in the
    # sampler's source contract; do not guess missing classes from Paddle.
    excluded = {"2ae5fe6d", "37c96354", "9ad49fcc", "d2dd2e5f", "361e7fed", "2418b6d9", "0ba3ad4f", "fd541ea7"}
    sources = [r['source'] for r in inventory if r['source'][4:12] not in excluded]
    priority = [r['source'] for r in inventory if r['amendments']]
    priority += [s for s in sources if config['capabilities'][s]['family'].startswith('dg') and s not in priority]
    priority += [s for s in sources if s not in priority]
    counts = {s: 2 + (s in priority[:20]) for s in sources}
    if batch > 1:
        # Retain all forty parents, but spend more draws on demonstrated name
        # continuations. Easier parents remain explicit controls, not credited
        # as difficult merely because of their source layout.
        counts = {s: 1 for s in sources}
        continuation_sources = priority[:14]
        if batch >= 5:
            observed = set()
            for result_path in sorted(OUT.glob('batch-*/native-review.json')):
                for row in read(result_path)['results']:
                    if any(o['kind']=='continuation_shared' for b in row['boundaries'] for o in b['occurrences']):
                        observed.add(row['source'])
            continuation_sources = [s for s in sources if s in observed]
        for s in continuation_sources:
            counts[s] = 4
        # A source with no accepted fresh continuation is not credited from a
        # parent-only screen. Keep it as a small fresh control/probe instead.
        if sum(counts.values()) > 100:
            raise ValueError('continuation weighting exceeds the approved batch size')
        preferred = [s for s in sources if config['capabilities'][s]['family'] != 'general']
        preferred += [s for s in priority if s not in preferred]
        while sum(counts.values()) < 100:
            for s in preferred:
                if counts[s] < 3 and sum(counts.values()) < 100:
                    counts[s] += 1
    assert len(sources) == 40 and sum(counts.values()) == 100
    selected_counts = {}
    for row in map(json.loads, (DATA / 'synthetic-train.jsonl').open()):
        sid = row['sourceDocumentId']
        if sid in sources:
            selected_counts[sid] = max(selected_counts.get(sid, 0), len(row['target'].get('containerInformation', [])))
    directory = OUT / f'batch-{batch:03d}'
    config.update(source_ids=sources, variants_per_source=max(counts.values()), seed=2026110900+batch,
                  template_sampling={'samples':100, 'source_counts':counts}, concurrency=8,
                  budget_usd=5, output=str((directory/'sampling').relative_to(ROOT)))
    config['capabilities'] = {s: {**config['capabilities'][s], 'destination_country':'AE'} for s in sources}
    config['declared_packages']['source_group_counts'] = {
        s: ([1]*counts[s] if s in config['declared_packages']['source_group_counts']
             and set(config['declared_packages']['source_group_counts'][s]) == {1}
             else [[1,2,1,3][(i+j+batch)%4] for j in range(counts[s])])
        for i,s in enumerate(sources)
    }
    # These three templates own a uniform aggregate equipment declaration,
    # not independent type cells. Retain their supported multi-container mode.
    uniform_aggregate = {'7b12b905', 'a53a1a9e', 'b5c11ec9'}
    mixed = [s for s in sources if selected_counts.get(s,0)>1 and s[4:12] not in uniform_aggregate]
    config['independent_preparation'].update(mixed_source_ids=mixed, mixed_probability=.5,
        thermal_goods_modes={s:'all' for s in sources if config['capabilities'][s]['family']=='chilled'})
    equipment = config['sampling'].get('equipment_inventory')
    if equipment and 'source_ids' in equipment:
        equipment['source_ids'] = [s for s in equipment['source_ids'] if s in sources]
    atomic_write_json(directory/'requested-config.json',config)
    readiness = inventory_sources(ROOT, config)
    atomic_write_json(directory/'source-readiness.json',readiness)
    print(json.dumps({k:readiness[k] for k in ('selectedSources','plannedSlots','readySources','counts')}),flush=True)
    for s,r in readiness['inventory'].items():
        if r['status']!='ready_source':print(s[4:12],r['reason'],flush=True)


async def generate(batch):
    directory = OUT / f'batch-{batch:03d}'
    config = read(directory/'requested-config.json')
    started = time.perf_counter()
    readiness = read(directory/'source-readiness.json')
    if readiness['readySlots'] != readiness['plannedSlots']:
        raise ValueError('source prerequisite holds require resolution before generation')
    prep_path = directory/'preparation.json'
    if prep_path.exists():
        prepared = read(prep_path)
    else:
        prepared = await prepare_campaign(ROOT, config, **config['independent_preparation'])
        # Resolve one-off invalid native outputs before freezing any candidate
        # or wording receipt. Replanning after rendering would invalidate later
        # adaptive category choices and force unnecessary lexical regeneration.
        invalid_states = {'held_native_choice','held_native_profile','held_declared_profile'}
        if config['independent_preparation'].get('retry_invalid_output') and any(
            count and state in invalid_states for state,count in prepared['counts'].items()
        ):
            atomic_write_json(directory/'preparation-first-attempt.json',prepared)
            prepared = await prepare_campaign(ROOT, config, **config['independent_preparation'])
        atomic_write_json(prep_path,prepared)
    print(json.dumps({'phase':'prepared','ready':prepared['readySlots'],'planned':prepared['plannedSlots'],'counts':prepared['counts']}),flush=True)
    if prepared['preparedConfig'] is None:
        raise ValueError('no ready shipments')
    campaign = create_campaign(ROOT,read(prepared['preparedConfig']['path']))
    generated = await campaign.generate_all()
    print(json.dumps({'phase':'wording','errors':generated['errors'],'costUsd':generated['costUsd']}),flush=True)
    postal = await campaign.correct_postal_all()
    rendered = campaign.render_all()
    atomic_write_json(directory/'generation.json',{'generation':generated,'postal':postal,'render':rendered,'seconds':time.perf_counter()-started})
    print(json.dumps({'phase':'candidates','count':len(rendered['accepted']),'errors':rendered['errors'],'seconds':time.perf_counter()-started}),flush=True)
    await native_propose(batch)


async def resume_preparation(batch):
    directory=OUT/f'batch-{batch:03d}'
    previous=read(directory/'preparation.json')
    archive=directory/'preparation-initial.json'
    if archive.exists():raise ValueError('the bounded preparation retry already ran')
    atomic_write_json(archive,previous)
    config=read(directory/'requested-config.json')
    prepared=await prepare_campaign(ROOT,config,**config['independent_preparation'])
    atomic_write_json(directory/'preparation.json',prepared)
    print(json.dumps({'ready':prepared['readySlots'],'counts':prepared['counts']}),flush=True)


async def repair_generation(batch):
    directory=OUT/f'batch-{batch:03d}'
    prepared=read(directory/'preparation.json')
    config=read(prepared['preparedConfig']['path'])
    campaign=create_campaign(ROOT,config)
    decisions=read(directory/'lexical-corrections.json')
    for prefix,changes in decisions.items():
        sid=next(s for s in campaign.source_ids if s[4:12]==prefix)
        receipts=[p for p in (campaign.output/'calls').glob('wording-validation-repair-*.json') if read(p)['identity']==sid]
        if len(receipts)!=1:raise ValueError('correction needs one exact prior call')
        prior=read(receipts[0]);payload=copy.deepcopy(prior['output'])
        for change in changes:
            if payload[change['shipment']][change['field']]!=change['before']:
                raise ValueError('reviewed lexical before-value differs')
            payload[change['shipment']][change['field']]=change['after']
        plans=list(campaign.plans(sid))
        requests=[campaign.wording_request(bp,scenario,identity,target) for bp,scenario,identity,target in plans]
        accepted=campaign._unpack_generated_wording(wording_output_type(requests).model_validate(payload),requests)
        for (bp,scenario,identity,target),request in zip(plans,requests,strict=True):
            path=campaign.output/'wording'/f'{identity}.json'
            if path.exists():raise ValueError('initial failed-generation correction cannot overwrite existing wording')
            atomic_write_json(path,{'sourceDocumentId':sid,'documentId':identity,
                'requestSha256':wording_request_hash(request),'scenario':scenario.model_dump(mode='json'),
                'targetBeforeWording':target,'values':accepted[identity],'blueprint':bp.inventory(),
                'manualCorrection':{'sourceCall':pin(receipts[0]),'decisions':pin(directory/'lexical-corrections.json')}})
    # Bounded retry of cached invalid output, preserving all successful receipts.
    campaign.calls.retry_invalid_output=True
    generated=await campaign.generate_all()
    postal=await campaign.correct_postal_all()
    rendered=campaign.render_all()
    atomic_write_json(directory/'generation-repair.json',{'generation':generated,'postal':postal,'render':rendered})
    print(json.dumps({'count':len(rendered['accepted']),'errors':[{**e,'error':e['error'][:250]} for e in rendered['errors']]}),flush=True)


async def native_propose(batch):
    directory = OUT / f'batch-{batch:03d}'
    prepared = read(directory/'preparation.json')
    config = read(prepared['preparedConfig']['path'])
    cfg = {k:config[k] for k in CuratedSynthesisConfig.model_fields}
    cfg['output'] = str((directory/'native-provider').relative_to(ROOT))
    runner = Runner(ROOT,cfg)
    cat = PaddleTemplateCatalog(ROOT,pin(CATALOG))
    candidates = sorted((ROOT/config['output']/'candidates').glob('*.json'))
    errors=[]
    async def one(path):
        try:
            spec = cat.wording_specification(path)
            if spec is None:
                return
            dest = directory/'wording-proposals'/f'{path.stem}.json'
            if dest.exists():
                if read(dest)['candidateSha256']!=sha256_file(path):
                    raise ValueError('existing wording belongs to different candidate')
                return
            result = await propose(runner,spec)
            atomic_write_json(dest,result)
        except Exception as exc:
            errors.append({'id':path.stem,'error':f'{type(exc).__name__}: {exc}'})
    for i in range(0,len(candidates),8):
        await asyncio.gather(*(one(p) for p in candidates[i:i+8]))
    receipt={'candidates':len(candidates),'errors':errors,'costUsd':str(runner.spent)}
    atomic_write_json(directory/'wording-proposals-summary.json',receipt)
    print(json.dumps(receipt),flush=True)


async def native_correct(batch):
    directory = OUT / f'batch-{batch:03d}'
    decisions=read(directory/'wording-corrections.json')
    prepared=read(directory/'preparation.json')
    config=read(prepared['preparedConfig']['path'])
    cfg={k:config[k] for k in CuratedSynthesisConfig.model_fields}
    cfg['output']=str((directory/'native-provider').relative_to(ROOT))
    runner=Runner(ROOT,cfg)
    cat=PaddleTemplateCatalog(ROOT,pin(CATALOG))
    for suffix,reason in decisions.items():
        paths=list((directory/'wording-proposals').glob(f'*{suffix}.json'))
        if len(paths)!=1:raise ValueError('correction must identify exactly one proposal')
        path=paths[0]
        previous=read(path)
        archived=directory/'wording-proposal-history'/path.name
        if not archived.exists():atomic_write_json(archived,previous)
        spec=cat.wording_specification(ROOT/config['output']/'candidates'/path.name)
        spec['context']['reviewCorrection']={'reason':reason,'previousValues':read(archived)['values']}
        result=await propose(runner,spec)
        result['correction']={'previous':pin(archived),'reason':reason}
        atomic_write_json(path,result)
        print(path.stem[-10:],json.dumps(result['values'],ensure_ascii=False),flush=True)


def approve_native(batch):
    directory=OUT/f'batch-{batch:03d}'
    decision=read(directory/'wording-review.json')
    paths=sorted((directory/'wording-proposals').glob('*.json'))
    digest=sha256_bytes(canonical_json_bytes([pin(p) for p in paths]))
    if len(paths)!=decision['count'] or digest!=decision['proposalsSha256']:
        raise ValueError('wording review does not cover this exact proposal set')
    for path in paths:
        value=read(path)
        for change in decision.get('valueCorrections',[]):
            if path.stem.endswith(change['suffix']):
                if value['values'][change['field']]!=change['before']:
                    raise ValueError('native wording correction before-value changed')
                value['values'][change['field']]=change['after']
                value.setdefault('manualCorrections',[]).append(change)
        value['status']='REVIEWED'
        value['review']=pin(directory/'wording-review.json')
        atomic_write_json(directory/'wording-reviewed'/path.name,value)
    print(json.dumps({'reviewedWording':len(paths)}))


def native_render(batch):
    directory = OUT / f'batch-{batch:03d}'
    prepared = read(directory/'preparation.json')
    config = read(prepared['preparedConfig']['path'])
    cat = PaddleTemplateCatalog(ROOT,pin(CATALOG))
    results,errors=[],[]
    for path in sorted((ROOT/config['output']/'candidates').glob('*.json')):
        try:
            spec=cat.wording_specification(path)
            wording=read(directory/'wording-reviewed'/f'{path.stem}.json') if spec is not None else None
            rendered=cat.render(path,wording_receipt=wording)
            candidate=read(path)
            template,_,_,_=cat._template(candidate['sourceDocumentId'])
            boundaries=boundary_readback(rendered,candidate,template['overlay'])
            dest=directory/'native'/f'{path.stem}.json'
            atomic_write_json(dest,rendered)
            results.append({'id':path.stem,'source':candidate['sourceDocumentId'], 'candidate':pin(path),
                            'rendered':pin(dest),'wording':pin(directory/'wording-reviewed'/f'{path.stem}.json') if wording else None,
                            'boundaries':boundaries,'goods':rendered['target'].get('goodsItemDetails'),
                            'parties':rendered['target'].get('parties'),
                            'containers':rendered['target'].get('containerInformation',[])})
        except Exception as exc:
            errors.append({'id':path.stem,'error':f'{type(exc).__name__}: {exc}'})
    atomic_write_json(directory/'native-review.json',{'results':results,'errors':errors,'catalog':pin(CATALOG)})
    counts=Counter(o['kind'] for r in results for b in r['boundaries'] for o in b['occurrences'])
    print(json.dumps({'count':len(results),'errors':errors,'boundaryOccurrences':dict(counts)}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "controls", "stage", "generate", "resume-preparation", "repair-generation", "native-propose", "native-correct", "approve-native", "native-render"])
    parser.add_argument('--batch',type=int,default=1)
    args = parser.parse_args()
    if args.action == 'stage':
        stage(args.batch)
    elif args.action == 'generate':
        asyncio.run(generate(args.batch))
    elif args.action == 'resume-preparation':
        asyncio.run(resume_preparation(args.batch))
    elif args.action == 'repair-generation':
        asyncio.run(repair_generation(args.batch))
    elif args.action == 'native-propose':
        asyncio.run(native_propose(args.batch))
    elif args.action == 'native-correct':
        asyncio.run(native_correct(args.batch))
    elif args.action == 'approve-native':
        approve_native(args.batch)
    elif args.action == 'native-render':
        native_render(args.batch)
    else:
        {"inventory": inventory, "controls": controls}[args.action]()
