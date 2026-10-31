"""Refresh only reviewed delivery-country pins; cargo ownership is unchanged."""

import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.independent_numeric import _digest
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign

REG = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed")
WORK = REG / "package-alias-rollout-20261030"
SID = "doc_ab905fd40d2afadc45840dc00b569e510bfd6a1017218d16c2ce125ea38288a7"


def main():
    review = json.loads((WORK / "delivery-country-repair/reviewed-patch.json").read_text())
    before, after = [copy.deepcopy(review[k]["target"]) for k in ("sourceBefore", "sourceAfter")]
    for target in (before, after):
        target["documentPatch"]["parties"].pop("deliveryAgent")
    assert before == after
    config = json.loads((WORK / "batches/native-v2/batch-01/config.json").read_text())
    pin = config["independent"]["numeric_certificates"]
    assert sha256_file(Path(pin["path"])) == pin["sha256"]
    data = json.loads(Path(pin["path"]).read_text())
    row = data["sources"][SID]
    config["source_ids"] = [SID]
    inventory = json.loads((WORK / "inspection/inventory.json").read_text())
    config["capabilities"] = {SID: inventory["config"]["capabilities"][SID]}
    blueprint = _SourceOwnershipCampaign(Path.cwd(), config).blueprint(SID)
    row["sourceContractSha256"] = _digest(blueprint.contract.model_dump(mode="json"))
    row["sourcePublicTargetSha256"] = _digest(review["sourceAfter"]["target"])
    row["decision"] += (
        " PDF-reviewed page-three delivery-country continuation added; "
        "cargo/measure owners unchanged."
    )
    atomic_publish_json(WORK / "prepared-v3/numeric_certificates-final.json", data)
    print(json.dumps({"source": SID, "cargoChanged": False, "sources": len(data["sources"])}))


if __name__ == "__main__":
    main()
