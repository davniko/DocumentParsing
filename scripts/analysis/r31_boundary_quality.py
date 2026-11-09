"""Owned final-line boundary measurements; these never change extraction labels."""

from __future__ import annotations

import re
import unicodedata

from document_ocr.labeling_agents.source_casing import selected_fields
from document_ocr.synthesis.paddle_source_casing import target_pointer


def compact(value):
    return "".join(c for c in unicodedata.normalize("NFKC", value).upper() if c.isalnum())


def owned_regions(rendered, candidate, overlay):
    paths = {f"{e['key']}@{e['byteStart']}:{e['byteEnd']}": set(map(target_pointer, e.get('targetPaths', [])))
             for e in candidate['proof']['edits']}
    for extra in overlay.get('extras', []):
        if 'target' in extra:
            paths.setdefault(extra['owner'], set()).add(extra['target'])
    refs = {}
    for r in rendered['proof']:
        for path in paths.get(r['owner'].split(':frame:')[0], []):
            refs.setdefault(path,set()).update((s['page'],s['region']) for s in r['spans'])
    for kind,selector,keys in [('targetFromOwners','keys',True),('targetFromRenderedOwners','owners',False)]:
        for action in overlay.get(kind,[]):
            refs[action['target']] = {(s['page'],s['region']) for r in rendered['proof']
                if (r['owner'].split('@')[0] if keys else r['owner'].split(':frame:')[0]) in action[selector]
                for s in r['spans']}
    for action in overlay.get('targetFromSourceSpans',[]):
        refs[action['target']] = {tuple(s[:2]) for g in action.get('groups',[action])
            for key in ('regions','fragments','spans') for s in g.get(key,[])}
    for path,positions in overlay.get('sourceCaseRegions',{}).items():
        refs[path] = set(map(tuple,positions))
    return refs


def boundary_readback(rendered, candidate, overlay):
    refs = owned_regions(rendered,candidate,overlay)
    fields = dict(selected_fields(rendered['target']))
    results = []
    for path,name in fields.items():
        if not path.endswith('/name') or path[:-4]+'addressLine' not in fields:
            continue
        address = fields[path[:-4]+'addressLine']
        owned = refs.get(path,set()) | refs.get(path[:-4]+'addressLine',set())
        fragments = []
        for page in rendered['geometry']['pages']:
            for block in page['blocks']:
                if any((page['page'],i) in owned for i in block['sourceRegions']):
                    fragments.extend({'page':page['page'],**f} for f in block['physicalFragments'])
        folded,origins = '',[]
        for i,f in enumerate(fragments):
            text = compact(f['text'])
            folded += text
            origins.extend([i]*len(text))
        n,a = compact(name),compact(address)
        names = list(re.finditer(re.escape(n),folded)) if n else []
        addresses = list(re.finditer(re.escape(a),folded)) if a else []
        matches=[]
        for nm in names:
            for am in addresses:
                if nm.end()>am.start() or folded[nm.end():am.start()] not in {'','ADD','ADDR','ADDRESS'}:
                    continue
                first,last,following = origins[nm.start()],origins[nm.end()-1],origins[am.start()]
                if fragments[first]['page']!=fragments[following]['page']:
                    continue
                kind = 'continuation_shared' if first!=last and last==following else 'whole_name_shared' if last==following else 'separate'
                matches.append({'kind':kind,'nameStart':fragments[first], 'boundary':fragments[last],
                                'addressStart':fragments[following]})
        results.append({'path':path.rsplit('/',1)[0], 'name':name,'address':address,
                        'occurrences':matches,'status':'matched' if matches else 'needs_review',
                        'ownedLines':[f['text'] for f in fragments] if not matches else []})
    return results
