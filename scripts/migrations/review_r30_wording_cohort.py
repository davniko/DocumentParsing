"""Record the main-agent readback for the explicitly inspected batches 2-4."""

from migrate_r30_synthetic import ART, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    for batch in (2, 3, 4):
        staged_path = OUT / "batches" / f"{batch:03d}-staging.json"
        staged = read(staged_path)
        retry_path = OUT / "batches" / f"{batch:03d}-wording-retries.json"
        retries = {read(verify_pin(p))["documentId"]: verify_pin(p)
                   for p in read(retry_path)["proposals"]}
        accepted = []
        for spec in staged["newWording"]:
            identity = spec["documentId"]
            path = retries.get(identity, ART / "wording-proposals" / (identity + ".json"))
            proposal = read(path)
            if proposal["candidateSha256"] != spec["candidateSha256"] or proposal["sourceFingerprint"] != spec["sourceFingerprint"]:
                raise ValueError("reviewed wording context changed")
            if set(proposal["values"]) != set(spec["requested"]):
                raise ValueError("wording fields differ from requested owners")
            proposal["status"] = "REVIEWED"
            proposal["review"] = {
                "proposal": pin(path), "staging": pin(staged_path),
                "scope": "Main agent read requested instructions, generated values and relevant party/locality context. Six one-off instruction/style deviations regenerated; input geometry and full document target review remain subsequent gates.",
            }
            if identity == "syn_full_v7_06856e9d8ef9aff286c35473":
                proposal["review"]["manualAddressCorrection"] = {
                    "before": proposal["values"]["forwarderAddress"],
                    "reason": "Keep the supplied TAPIS TERMINAL locality; do not add Dungun/Terengganu as new public address geography.",
                }
                proposal["values"]["forwarderAddress"] = "LOT 7, JALAN PELABUHAN\nTAPIS TERMINAL\n24000"
            output = ART / "wording" / (identity + ".json")
            atomic_publish_json(output, proposal)
            accepted.append(pin(output))
        atomic_publish_json(OUT / "batches" / f"{batch:03d}-wording-review.json", {"accepted": accepted})
        print(batch, len(accepted))


if __name__ == "__main__":
    main()
