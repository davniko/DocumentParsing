"""Resume an interrupted batch in a new directory, reusing exact completed calls.

Never rewrite old outputs. Reuse requires identical source, prompt, schema,
context, model, effort and attachment hashes. Failed/in-flight calls are not
accepted and remain explicitly accounted for in the recovery inventory.
"""
import argparse
import asyncio
import json
import shutil
from pathlib import Path

import run_real_v7_batch as runner
from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_bytes, sha256_file


def recover(source, dest):
    assert not dest.exists()
    manifest = json.loads((source / 'selection.json').read_text())
    runner.verify(manifest)
    cache, inventory = {}, []
    for path in sorted(source.glob('runs/*/*/calls/*-request.json')):
        request = json.loads(path.read_text())
        result_path = path.with_name(path.name.replace('-request', '-result'))
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        status = result['status'] if result else 'interrupted_no_receipt'
        inventory.append(dict(request=str(path.relative_to(runner.ROOT)), status=status,
            estimatedUsd=runner.usage(result)[1] if result else None,
            billingStatus=result.get('billingStatus') if result else 'unknown'))
        if status == 'completed':
            key = json.dumps(request, sort_keys=True, separators=(',', ':'))
            assert key not in cache, 'Ambiguous multiple successful responses to identical request'
            cache[key] = (result_path, result)
    dest.mkdir(parents=True)
    shutil.copy2(source / 'config.json', dest / 'config.json')
    manifest['sources'][str((source / 'selection.json').relative_to(runner.ROOT))] = sha256_file(source / 'selection.json')
    manifest['implementation'][str(Path(__file__).resolve().relative_to(runner.ROOT))] = sha256_file(Path(__file__))
    atomic_publish_json(dest / 'selection.json', manifest)
    atomic_publish_json(dest / 'recovery-inventory.json', dict(source=str(source.relative_to(runner.ROOT)), calls=inventory))
    base = runner.DirectLabelingFlow

    class ReusingFlow(base):
        async def _call(self, stage, output_model, *, context='', attachments=None, complete_pdf=False, audit=False):
            request = dict(stage=stage, model=self.config.model, outputMode='native_json_schema', strict=True,
                reasoningEffort=self.config.audit_reasoning_effort if audit and self.config.audit_reasoning_effort is not None else self.config.reasoning_effort,
                ocrSha256=sha256_bytes(self.ocr.encode()), promptSha256=sha256_bytes(self.prompts[stage].encode()),
                schema=self.output_schema(output_model), context=context,
                imageSha256=[sha256_bytes(b) for b in attachments or []],
                completePdfSha256=sha256_file(self.pdf_path) if complete_pdf else None)
            key = json.dumps(request, sort_keys=True, separators=(',', ':'))
            if key not in cache:
                return await super()._call(stage, output_model, context=context, attachments=attachments,
                    complete_pdf=complete_pdf, audit=audit)
            old_path, old = cache[key]
            answer = output_model.model_validate_json(json.dumps(old['answer']))
            self._check_package_tokens(answer.model_dump(mode='json'))
            self._sequence += 1
            number = self._sequence
            receipt = dict(call=number, stage=stage, status='completed', answer=answer.model_dump(mode='json'),
                responses=[], elapsedSeconds=0, billingStatus='reused_without_request',
                reusedReceipt=str(old_path.relative_to(runner.ROOT)), reusedReceiptSha256=sha256_file(old_path))
            atomic_publish_json(self.output_dir / 'calls' / f'{number:03d}-request.json', request)
            atomic_publish_json(self.output_dir / 'calls' / f'{number:03d}-result.json', receipt)
            self.receipts.append(receipt)
            return answer

    runner.DirectLabelingFlow = ReusingFlow
    asyncio.run(runner.run(dest))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--dest', required=True, type=Path)
    a = p.parse_args()
    recover(runner.ROOT / a.source, runner.ROOT / a.dest)
