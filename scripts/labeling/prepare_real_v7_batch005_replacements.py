"""Four source-reviewed fresh shipments replacing test/duplicate wrappers."""
import json
from run_real_v7_batch import ROOT, prepare
from document_ocr.atomic import atomic_publish_json

def main():
    source=ROOT/'artifacts/kie-labeling/topup-screen-20261005/inventory.json'
    inventory=json.loads(source.read_text())
    prefixes={'00f07a75','06a7c8b2','08ec8a5f','0abced73'}
    selected=[r for r in inventory['documents'] if r['documentId'][4:12] in prefixes]
    assert len(selected)==4 and all(not r['exclusions'] and r['split']=='train' and r['pages']<=5 for r in selected)
    inventory['documents']=selected
    inventory['purpose']='Replacements for batch005 test/duplicate-shipment sources 068,087,111,136; all four OCR sources manually screened before model calls.'
    path=source.with_name('batch005-replacement-inventory.json')
    atomic_publish_json(path,inventory)
    prepare(ROOT/'artifacts/kie-labeling/direct-real-batch005-replacements-20261005',train=4,validation=0,seed=2026100506,inventory=path)

if __name__=='__main__': main()
