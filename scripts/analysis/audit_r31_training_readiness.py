"""Read-only R31 preflight: recomputed printed case and rendered boundary diversity.

Evidence files are read, not regenerated. Findings are review candidates, never
permission to rewrite labels automatically.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import resource
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.labeling_agents.source_casing import PrintedCaseIndex, recover_case, selected_fields
from document_ocr.spatial_inputs.paddle import pages_from_record, serialize, text_view_digests
from document_ocr.synthesis.paddle_source_casing import POLICY, rendered_case_scopes
from scripts.analysis.r31_boundary_quality import boundary_readback, compact
from scripts.migrations.migrate_r31_casing import native_scopes, reviewed_overlay, reviews, validate_case_only

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/curated/mpci-bl-r31-paddle-centroids-v1'
OUT = ROOT / 'docs/analysis/r31-final-readiness-20261109'
R30 = ROOT / 'data/curated/mpci-bl-r30-paddle-centroids-v1'


def read(path):
    return json.loads((ROOT / path).read_text())


@lru_cache(maxsize=512)
def cached(path):
    return read(path)


def native_evidence(row):
    e = row['evidence']
    for key in ('candidate', 'rendered'):
        assert sha256_file(ROOT / e[key]['path']) == e[key]['sha256']
    rendered = read(e['rendered']['path'])
    assert rendered['pages'] == row['pages'], 'render evidence does not describe current input'
    candidate = read(e['candidate']['path'])
    if candidate.get('status') == 'REVIEWED_SOURCE_PRESERVING_REBASE':
        return rendered, candidate, None, None
    cat = cached(e['catalog']['path'])
    template = cached(cat['templates'][row['sourceDocumentId']]['template']['path'])
    overlay = template['overlay']
    if cat.get('targetCasing') != POLICY:
        overlay = reviewed_overlay(overlay, row['sourceDocumentId'][4:12])
    return rendered, candidate, overlay, cached(template['source']['path'])


def scan_casing():
    started = time.perf_counter()
    manifest = read(DATA / 'manifest.json')
    result, findings, flags, occurrence_examples = {}, [], [], []
    ids = set()
    for split in ('validation', 'real-train', 'synthetic-train'):
        assert sha256_file(DATA / f'{split}.jsonl') == manifest['aggregateRecords'][split]['sha256']
        old_rows = iter(map(json.loads, (R30 / f'{split}.jsonl').open()))
        counts, fields, methods = Counter(), defaultdict(Counter), Counter()
        for index, line in enumerate((DATA / f'{split}.jsonl').open()):
            row = json.loads(line); identity = row['documentId']
            assert identity not in ids; ids.add(identity)
            old = next(old_rows, None)
            if old is not None:
                assert identity == old['documentId']
                compared = row
                if 'agencyDiversityAmendment' in row or 'readinessDescriptionAmendment' in row:
                    compared = read(row['evidence']['previousRecord']['path'])
                    counts['explicitSuffixAmendments' if 'agencyDiversityAmendment' in row else 'explicitDescriptionAmendments'] += 1
                validate_case_only(old['target'], compared['target'])
                assert old['pages'] == compared['pages']
                counts['r30CaseOnlyComparisons'] += 1
            assert BillOfLadingExtractionV7DirectTarget.model_validate_json(json.dumps(row['target'])).canonical_target() == row['target']
            pages = pages_from_record(row)
            for mode, field in [('none', 'joinedRawText'), ('centroid', 'positionedText'), ('bbox', 'bboxText')]:
                assert serialize(pages, geometry=mode) == row[field]
            if split == 'synthetic-train':
                assert all(row[k] == v for k, v in text_view_digests(row).items())
                rendered, candidate, overlay, source = native_evidence(row)
                scopes = rendered_case_scopes(rendered, candidate, overlay, source) if overlay is not None else {}
            else:
                scopes = native_scopes(row)
            if split != 'synthetic-train' or overlay is None:
                for path, refs in reviews()['recordRegions'].get(row['sourceDocumentId'][4:12], {}).items():
                    scopes[path] = PrintedCaseIndex('\n'.join(row['pages'][p]['regions'][i]['text'] for p, i in refs))
            text = '\n'.join(r['text'] for p in row['pages'] for r in p['regions'])
            printed = PrintedCaseIndex(text)
            counts['documents'] += 1
            counts['regions'] += sum(len(p['regions']) for p in row['pages'])
            for path, value in selected_fields(row['target']):
                kind = path.rsplit('/', 1)[-1]; stat = fields[kind]
                stat['total'] += 1
                stat['uppercase' if value.isupper() else 'lowercase' if value.islower() else 'mixed'] += 1
                stat['hasNewline'] += '\n' in value or '\r' in value
                stat['surroundingWhitespace'] += value != value.strip()
                scope = scopes.get(path)
                # Deliberately start from uppercase, so agreement is not merely
                # retaining an already mixed-case input target.
                recovered = recover_case(value.upper(), printed, owned=scope)
                explicit = reviews().get('recordUppercase', {}).get(identity[4:12], {}).get(path)
                if explicit:
                    assert scope is not None and scope.text.isupper() and value.isupper()
                    recovered = {'value': value, 'method': 'reviewed_uppercase_continuation'}
                methods[recovered['method']] += 1
                if recovered.get('value') != value:
                    findings.append({'split': split, 'id': identity, 'path': path, 'value': value,
                                     'ownedText': scope.text if scope else None, 'recovered': recovered})
                variants = printed.variants(value, word_edges=False)
                stat['literalIgnoringPhysicalWhitespace'] += value in variants
                stat['multiplePrintedCases'] += len(variants) > 1
                if len(variants) > 1 and len(occurrence_examples) < 80:
                    occurrence_examples.append({'split': split, 'id': identity, 'path': path, 'value': value,
                                                'variants': variants, 'ownedText': scope.text if scope else None})
                suspect = []
                if '\n' in value or '\r' in value: suspect.append('target_newline')
                if re.search(r'(?i)\b(?:fictional|placeholder|your output|generate a|do not repeat)\b', value):
                    suspect.append('instruction_or_placeholder_words')
                words = value.casefold().split()
                if any(words[i:i+3] == words[i+3:i+6] for i in range(len(words)-5)):
                    suspect.append('consecutive_repeated_three_word_phrase')
                if kind == 'addressLine':
                    role = row['target']
                    for part in path.strip('/').split('/')[:-1]:
                        role = role[int(part)] if isinstance(role, list) else role[part]
                    name = role.get('name', '')
                    if len(name) >= 12 and compact(name) in compact(value):
                        suspect.append('complete_company_name_inside_address')
                if suspect:
                    flags.append({'split': split, 'id': identity, 'parent': row['sourceDocumentId'],
                                  'path': path, 'value': value, 'flags': suspect})
            if counts['documents'] % 500 == 0:
                print(split, counts['documents'], 'case findings', len(findings), flush=True)
        result[split] = {'counts': dict(counts), 'fields': {k: dict(v) for k, v in fields.items()}, 'methods': dict(methods)}
    receipt = {'status': 'REVIEW' if findings or flags else 'PASS', 'splits': result,
               'caseFindings': findings, 'outlierScreens': flags, 'occurrenceExamples': occurrence_examples,
               'seconds': time.perf_counter()-started, 'peakRssKiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'manifestSha256': sha256_file(DATA / 'manifest.json'), 'uniqueRecords': len(ids)}
    atomic_write_json(OUT / 'casing-audit.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('caseFindings','outlierScreens','occurrenceExamples')}))
    print('caseFindings', len(findings), 'outlierScreens', len(flags))


def describe_boundary(boundary):
    name, address = boundary['name'], boundary['address']
    occurrence = next((o for o in boundary['occurrences'] if o['kind'] == 'continuation_shared'),
                      next((o for o in boundary['occurrences'] if o['kind'] == 'whole_name_shared'), None))
    if occurrence is None:
        return None
    text = occurrence['boundary']['text']
    # Locate the beginning of the address within the final line using all the
    # available address prefix, not an unowned short anchor elsewhere in the bill.
    folded = ''; origins = []
    for i, char in enumerate(text):
        part = compact(char); folded += part; origins.extend([i]*len(part))
    addr = compact(address)
    # Full names/addresses have already been matched within owned fragments.
    # A boundary line may contain a caption before the name, only the first
    # "P.O." of the address, or a phone after the complete address. Do not
    # require a minimum-length address anchor or invent a boundary from one.
    name_key = compact(name)
    matches = [i for i in range(1, len(folded))
               if (addr.startswith(folded[i:]) or folded[i:].startswith(addr))
               and (name_key.endswith(folded[:i]) or folded[:i].endswith(name_key))]
    if len(matches) != 1:
        return {'kind': occurrence['kind'], 'classification': 'needs_junction_review', 'line': text}
    at = matches[0]
    junction = text[origins[at-1]+1:origins[at]]
    if not junction: delimiter = 'no separator'
    elif junction.isspace(): delimiter = 'whitespace only'
    elif any(c.isspace() for c in junction): delimiter = 'punctuation and whitespace'
    else: delimiter = 'punctuation without whitespace'
    if re.search(r'\([^()]+\)\s*$', name): ending = 'parenthesized qualifier'
    elif re.search(r'(?i)\b(?:L\.?L\.?C\.?|LTD\.?|LIMITED|GMBH|S\.?A\.?R\.?L\.?|S\.?R\.?L\.?|S\.?A\.?|B\.?V\.?|CO\.?|INC\.?|FZCO|FZE|FZC|FZ|PLC|SAS|AS|LDA|LTDA|OY|AB|DMCC|EST\.?|ESTABLISHMENT|CORP\.?|KG|AG|CC|SPA|SNC)\s*$', name): ending = 'legal/business suffix'
    elif re.search(r'(?i)\b(?:AGENCY|AGENCIES|TRADING|TRADE|EXPORT|IMPORT|SERVICES|SUPPLIES|LOGISTICS|TRANSPORT|SHIPPING|FORWARDING|HANDLING|GROUP|INDUSTRIES|COMPANY|STORES|COURIER|SOLUTIONS)\W*$', name): ending = 'business word'
    else: ending = 'other name ending'
    if re.match(r'(?i)\s*P\.?\s*O\.?\s*BOX', address): lead = 'PO box'
    elif re.match(r'(?i)\s*C\s*/\s*O\b', address): lead = 'care of'
    elif re.match(r'\s*\d', address): lead = 'number-led'
    else: lead = 'word-led'
    return {'kind': occurrence['kind'], 'classification': 'matched', 'delimiter': delimiter,
            'ending': ending, 'addressStart': lead, 'line': text, 'junction': junction,
            'nameStartLine': occurrence['nameStart']['text']}


def scan_boundaries():
    started = time.perf_counter(); rows = []; stats = defaultdict(Counter); parents = defaultdict(set)
    new_docs = Counter(); paired = 0; unique_boundaries = set(); new_parents = set()
    for i, line in enumerate((DATA / 'synthetic-train.jsonl').open()):
        row = json.loads(line)
        if i < 9400:
            continue
        new_parents.add(row['sourceDocumentId'])
        rendered, candidate, overlay, _ = native_evidence(row)
        assert overlay is not None
        rendered = dict(rendered, target=row['target'])
        results = boundary_readback(rendered, candidate, overlay)
        paired += len(results); doc_kinds = set()
        for boundary in results:
            detail = describe_boundary(boundary)
            if detail is None:
                stats['role_status']['separate' if boundary['occurrences'] else 'unresolved'] += 1
                continue
            entry = {'id': row['documentId'], 'parent': row['sourceDocumentId'], 'role': boundary['path'],
                     'name': boundary['name'], 'address': boundary['address'], **detail}
            rows.append(entry); kind = detail['kind']; doc_kinds.add(kind)
            parents[kind].add(row['sourceDocumentId']); stats['role_status'][kind] += 1
            stats['roles'][boundary['path']] += 1
            for key in ('delimiter', 'ending', 'addressStart', 'classification'):
                stats[key][detail.get(key, 'unclassified')] += 1
                stats[f'{kind}:{key}'][detail.get(key, 'unclassified')] += 1
            stats[f'{kind}:last_word'][re.findall(r'\w+', boundary['name'])[-1].upper()] += 1
            unique_boundaries.add((row['documentId'], boundary['name'], boundary['address'], detail['line']))
        new_docs['continuation' if 'continuation_shared' in doc_kinds else 'whole' if doc_kinds else 'control_or_unresolved'] += 1
    result = {'newDocuments': dict(new_docs), 'newParents': len(new_parents), 'pairedRoles': paired,
              'parentCounts': {k: len(v) for k, v in parents.items()}, 'uniqueDocumentNameAddressLines': len(unique_boundaries),
              'statistics': {k: dict(v) for k, v in stats.items()}, 'rows': rows,
              'seconds': time.perf_counter()-started, 'peakRssKiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    atomic_write_json(OUT / 'boundary-audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['casing', 'boundaries'])
    args = parser.parse_args()
    {'casing': scan_casing, 'boundaries': scan_boundaries}[args.mode]()
