import json
from pathlib import Path

import pytest

from document_ocr.atomic import atomic_write_json
from scripts.synthesis import publish_r31_extension as publication


def test_publication_requires_review_and_preserves_stage_a(tmp_path, monkeypatch):
    data=tmp_path/'dataset';out=tmp_path/'campaign';catalog=out/'catalog.json'
    data.mkdir();out.mkdir()
    monkeypatch.setattr(publication,'ROOT',tmp_path)
    monkeypatch.setattr(publication,'DATA',data)
    monkeypatch.setattr(publication,'OUT',out)
    monkeypatch.setattr(publication,'CATALOG',catalog)
    def read(path):return json.loads((tmp_path/path).read_text())
    def pin(path):return {'path':str(path.relative_to(tmp_path)),'sha256':publication.sha256_file(path)}
    monkeypatch.setattr(publication,'read',read);monkeypatch.setattr(publication,'pin',pin)
    atomic_write_json(catalog,{'targetCasing':'printed-party-and-goods-v1'})
    for split,n in [('real-train',600),('synthetic-train',9400),('validation',60)]:
        (data/f'{split}.jsonl').write_text(''.join(json.dumps({'documentId':f'{split}-{i}'})+'\n' for i in range(n)))
    before=(data/'synthetic-train.jsonl').read_bytes()
    atomic_write_json(data/'manifest.json',{'stage':'A-casing-only','batches':[],
        'aggregateRecords':{s:pin(data/f'{s}.jsonl') for s in ['real-train','synthetic-train','validation']}})
    with pytest.raises(ValueError,match='only 0 reviewed'):publication.publish(100)
    batch=out/'batch-001';batch.mkdir();results=[]
    for i in range(100):
        identity=f'new-{i}'
        p=batch/f'{identity}.json'
        atomic_write_json(p,{'documentId':identity,'sourceDocumentId':'parent','target':{},'pages':[],
            'joinedRawText':'text','positionedText':'text || 1,2','bboxText':'text || 0,1,2,3','sourceCasing':{}})
        results.append({'id':identity,'candidate':pin(p),'rendered':pin(p),'wording':None})
    atomic_write_json(batch/'native.json',{'results':results,'catalog':pin(catalog)})
    atomic_write_json(batch/'audit.json',{'count':100})
    decision={'status':'MAIN_AGENT_REVIEWED','nativeReview':pin(batch/'native.json'),
              'audit':pin(batch/'audit.json'),'accepted':[r['id'] for r in results]}
    atomic_write_json(batch/'acceptance.json',decision)
    publication.publish(100)
    assert (data/'synthetic-train.jsonl').read_bytes().startswith(before)
    assert len((data/'synthetic-train.jsonl').read_text().splitlines())==9500
    assert read(data/'manifest.json')['stageB']['addedSynthetic']==100
    with pytest.raises(ValueError,match='only 0 reviewed'):publication.publish(100)
    atomic_write_json(batch/'audit.json',{'count':99})
    with pytest.raises(ValueError,match='artifact changed'):publication.publish(100)


def test_publication_rejects_arbitrary_batch_size():
    with pytest.raises(ValueError,match='final 50'):publication.publish(99)
