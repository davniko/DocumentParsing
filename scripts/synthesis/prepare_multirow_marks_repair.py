"""Bind the reviewed repeated consignee locality/country in source-only Marks."""

import copy
import json
import re
from pathlib import Path

import yaml

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign


def replace_source(content, sid, value):
    block = (
        "\n".join(
            "  " + line if line else ""
            for line in yaml.safe_dump(
                {sid: value}, allow_unicode=True, sort_keys=False, width=100
            ).splitlines()
        )
        + "\n"
    )
    content, count = re.subn(
        r"^  " + re.escape(sid) + r":\n.*?(?=^  doc_|\Z)",
        lambda _: block,
        content,
        flags=re.M | re.S,
    )
    if count != 1:
        raise ValueError("source authority block is not unique")
    return content


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    config = json.loads(
        Path("configs/synthesis/mpci_bl_curated_v7_package_aliases.json").read_text()
    )
    sid = next(s for s in config["source_ids"] if s.startswith("doc_551f3a34"))
    config["source_ids"] = [sid]
    config["capabilities"] = {sid: config["capabilities"][sid]}
    before = _SourceOwnershipCampaign(Path.cwd(), config).blueprint(sid)
    clause = "ALEXANDRIA\n(S.A.E.)EGYPT"
    if before.source.count(clause) != 1:
        raise ValueError("reviewed Marks clause changed")
    start = before.source.index(clause)
    ownership = root / "authority-repairs/locality-frames-expanded/ownership.yaml"
    auxiliary = Path(config["auxiliary"])
    own_text, aux_text = ownership.read_text(), auxiliary.read_text()
    own, aux = yaml.safe_load(own_text)["sources"][sid], yaml.safe_load(aux_text)["sources"][sid]
    decision = []
    for field, token in (("name", "ALEXANDRIA"), ("country", "EGYPT")):
        offset = start + clause.index(token)
        starts = [m.start() for m in re.finditer(re.escape(token), before.source)]
        key = "reviewed_marks_consignee_" + field
        own.setdefault("add", []).append(
            {
                "key": key,
                "kind": "location",
                "paths": [],
                "occurrences": [{"text": token, "occurrence": starts.index(offset) + 1}],
            }
        )
        aux.setdefault("bindings", {})[key] = {
            "party_location": "documentPatch.parties.consignee",
            "field": field,
        }
        decision.append(
            {
                "key": key,
                "sourceText": token,
                "occurrence": starts.index(offset) + 1,
                "field": field,
            }
        )
    output = root / "authority-repairs/marks-and-localities"
    new_own, new_aux = output / "ownership.yaml", output / "auxiliary.yaml"
    atomic_publish_bytes(new_own, replace_source(own_text, sid, own).encode())
    atomic_publish_bytes(new_aux, replace_source(aux_text, sid, aux).encode())
    after_config = copy.deepcopy(config)
    after_config.update(ownership=str(new_own), auxiliary=str(new_aux))
    after = _SourceOwnershipCampaign(Path.cwd(), after_config).blueprint(sid)
    if before.source != after.source or before.target != after.target:
        raise ValueError("Marks binding repair altered source bytes or target semantics")
    atomic_publish_json(
        output / "review.json",
        {
            "status": "COMPILED_REQUIRES_FRESH_RENDER",
            "sourceId": sid,
            "original": {"ownership": sha256_file(ownership), "auxiliary": sha256_file(auxiliary)},
            "bindings": decision,
            "sourceAndTargetUnchanged": True,
            "retainedBusinessSuffix": (
                "(S.A.E.) is a legal-form suffix, not an independently sampled location"
            ),
        },
    )
    print(json.dumps({"sources": 1, "bindings": len(decision), "sourceAndTargetUnchanged": True}))


if __name__ == "__main__":
    main()
