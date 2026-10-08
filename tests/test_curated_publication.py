import json
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_publication import publish_campaign, review_contract_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def campaign(tmp_path):
    dataset = tmp_path / "source"
    dataset.mkdir()
    rows = {
        sid: {"documentId": sid, "joinedRawText": "SOURCE " + sid, "target": {"source": sid}}
        for sid in ("source_a", "source_b")
    }
    (dataset / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows.values()))
    (dataset / "validation.jsonl").write_text(
        json.dumps({"documentId": "heldout", "joinedRawText": "HELD OUT", "target": {}}) + "\n"
    )
    constraints = tmp_path / "task.json"
    write(constraints, {"task": "current_target", "targetSchemaSha256": "a" * 64})
    output = tmp_path / "pilot"
    config = {
        "dataset": "source",
        "source_ids": list(rows),
        "variants_per_source": 2,
        "seed": 42,
        "task_constraints": "task.json",
        "task_constraints_sha256": digest(constraints.read_bytes()),
    }
    expected, replays = {}, []
    for sid in rows:
        candidates = {}
        for variant in (1, 2):
            sample_id = "syn_full_v7_" + digest([sid, 42, variant])[:24]
            text = f"NEW SHIPMENT {sid} {variant}\nCOUNTRY NETHERLANDS\n``` RAW BACKTICKS"
            candidate = {
                "documentId": sample_id,
                "sourceDocumentId": sid,
                "variant": variant,
                "seed": 42,
                "joinedRawText": text,
                "joinedRawTextSha256": digest(text.encode()),
                "target": {"addressLine": "COUNTRY NETHERLANDS", "quantity": variant},
                "proof": {"text": digest(text.encode())},
            }
            write(output / "candidates" / f"{sample_id}.json", candidate)
            expected[sample_id] = deepcopy(candidate)
            candidates[sample_id] = digest(candidate)
        write(
            output / "reviews" / f"{sid}.json",
            {
                "sourceDocumentId": sid,
                "candidateHashes": candidates,
                "reviewContractSha256": review_contract_hash(2),
                "review": {"reviewed_ids": list(candidates), "findings": []},
            },
        )

    def replay(candidate):
        replays.append(candidate["documentId"])
        if candidate != expected[candidate["documentId"]]:
            raise ValueError("candidate differs from authority replay")

    return SimpleNamespace(
        root=tmp_path,
        output=output,
        config=config,
        rows=rows,
        calls=SimpleNamespace(spent=Decimal("0.0123")),
        replay_candidate=replay,
        expected=expected,
        replays=replays,
    )


def add_finding(campaign, *, adjudicated=False):
    sid = campaign.config["source_ids"][0]
    review_path = campaign.output / "reviews" / f"{sid}.json"
    receipt = json.loads(review_path.read_text())
    sample_id = receipt["review"]["reviewed_ids"][0]
    finding = {
        "sample_id": sample_id,
        "field": "addressLine",
        "problem": "Country is allegedly absent",
        "evidence": "COUNTRY NETHERLANDS",
        "correction": "Add the country",
    }
    receipt["review"]["findings"] = [finding]
    write(review_path, receipt)
    approval = {
        "sourceDocumentId": sid,
        "reviewSha256": digest(receipt),
        "candidateHashes": receipt["candidateHashes"],
        "decisions": [
            {
                "findingIndex": 0,
                "findingSha256": digest(finding),
                "decision": "rejected",
                "reason": "The complete country is already printed and retained in the target.",
                "evidence": ["COUNTRY NETHERLANDS"],
            }
        ],
    }
    adjudication_path = campaign.output / "adjudications" / f"{sid}.json"
    if adjudicated:
        write(adjudication_path, approval)
    return adjudication_path, approval


def test_complete_publication_replays_every_candidate_and_preserves_sources(campaign):
    before = {path.name: path.read_bytes() for path in (campaign.root / "source").glob("*.jsonl")}
    add_finding(campaign, adjudicated=True)
    result = publish_campaign(campaign)
    assert result["published"] and result["valid"] == 4
    assert set(campaign.replays) == set(campaign.expected)
    assert len(campaign.replays) == len(campaign.expected)
    manifest = json.loads((campaign.output / "manifest.json").read_text())
    assert manifest["unresolvedReviewFindings"] == 0
    assert manifest["rejectedReviewFindings"] == 1
    assert manifest["reviewContractSha256"] == review_contract_hash(2)
    for filename, expected in manifest["files"].items():
        assert digest((campaign.output / filename).read_bytes()) == expected
    records = list(map(json.loads, (campaign.output / "dataset.jsonl").read_text().splitlines()))
    assert len(records) == 4
    assert all(
        record["target"] == campaign.expected[record["documentId"]]["target"] for record in records
    )
    assert all("inputText" not in record for record in records)
    gallery = (campaign.output / "samples.md").read_text()
    assert "````text" in gallery
    assert gallery.count("### Variant") == 4
    assert gallery.index("SOURCE source_a") < gallery.index("NEW SHIPMENT source_a 1")
    assert before == {
        path.name: path.read_bytes() for path in (campaign.root / "source").glob("*.jsonl")
    }
    assert publish_campaign(campaign) == result


def test_publication_uses_exact_nonuniform_template_quotas(campaign):
    for index, row in enumerate(campaign.rows.values()):
        row["target"] = {
            "documentPatch": {"negotiability": "negotiable" if index == 0 else "non_negotiable"}
        }
    source = campaign.root / "source/train.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in campaign.rows.values()))
    campaign.config["template_sampling"] = {"samples": 3, "negotiable_fraction": 2 / 3}
    sid = "source_b"
    removed = "syn_full_v7_" + digest([sid, 42, 2])[:24]
    (campaign.output / "candidates" / f"{removed}.json").unlink()
    review_path = campaign.output / "reviews" / f"{sid}.json"
    review = json.loads(review_path.read_text())
    review["candidateHashes"].pop(removed)
    review["review"]["reviewed_ids"].remove(removed)
    review["reviewContractSha256"] = review_contract_hash(1)
    write(review_path, review)
    result = publish_campaign(campaign)
    assert result["expected"] == result["valid"] == 3
    manifest = json.loads((campaign.output / "manifest.json").read_text())
    assert manifest["variantsPerSource"] == {"source_a": 2, "source_b": 1}
    assert manifest["reviewContractSha256BySource"][sid] == review_contract_hash(1)


@pytest.mark.parametrize("kind", ["candidate", "review", "adjudication"])
def test_missing_prerequisite_cannot_publish_partial_dataset(campaign, kind):
    if kind == "adjudication":
        add_finding(campaign)
    else:
        directory = campaign.output / ("candidates" if kind == "candidate" else "reviews")
        next(directory.glob("*.json")).unlink()
    with pytest.raises(ValueError, match="missing publication prerequisite"):
        publish_campaign(campaign)
    assert not (campaign.output / "dataset.jsonl").exists()
    assert not (campaign.output / "manifest.json").exists()


def test_self_consistent_candidate_and_review_tampering_fails_authority_replay(campaign):
    sample_id, candidate = next(iter(campaign.expected.items()))
    altered = deepcopy(candidate)
    altered["target"]["quantity"] = 999
    altered["joinedRawText"] += "\n999 PACKAGES"
    altered["joinedRawTextSha256"] = digest(altered["joinedRawText"].encode())
    altered["proof"]["text"] = altered["joinedRawTextSha256"]
    write(campaign.output / "candidates" / f"{sample_id}.json", altered)
    path = campaign.output / "reviews" / f"{candidate['sourceDocumentId']}.json"
    receipt = json.loads(path.read_text())
    receipt["candidateHashes"][sample_id] = digest(altered)
    write(path, receipt)
    with pytest.raises(ValueError, match="authority replay"):
        publish_campaign(campaign)
    assert not (campaign.output / "dataset.jsonl").exists()


def test_stale_review_cannot_approve_a_current_candidate(campaign):
    path = next((campaign.output / "reviews").glob("*.json"))
    receipt = json.loads(path.read_text())
    receipt["candidateHashes"][next(iter(receipt["candidateHashes"]))] = "0" * 64
    write(path, receipt)
    with pytest.raises(ValueError, match="stale review"):
        publish_campaign(campaign)


@pytest.mark.parametrize("missing", [True, False])
def test_review_under_missing_or_different_instructions_cannot_publish(
    campaign, monkeypatch, missing
):
    path = next((campaign.output / "reviews").glob("*.json"))
    receipt = json.loads(path.read_text())
    if missing:
        del receipt["reviewContractSha256"]
    else:
        monkeypatch.setattr(
            "document_ocr.synthesis.curated_wording.RENDERED_REVIEW_PROMPT",
            "A revised semantic review policy.",
        )
    write(path, receipt)
    with pytest.raises(ValueError, match="review instruction/schema contract"):
        publish_campaign(campaign)
    assert not (campaign.output / "dataset.jsonl").exists()
    assert not (campaign.output / "manifest.json").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("empty", "unresolved"),
        ("fixed", "new review"),
        ("stale", "stale adjudication"),
        ("evidence", "exact rendered evidence"),
        ("finding", "another finding"),
    ],
)
def test_adjudication_must_resolve_exact_finding_without_invented_evidence(
    campaign, change, message
):
    path, approval = add_finding(campaign)
    if change == "empty":
        approval["decisions"] = []
    elif change == "fixed":
        approval["decisions"][0]["decision"] = "fixed"
    elif change == "stale":
        approval["reviewSha256"] = "0" * 64
    elif change == "evidence":
        approval["decisions"][0]["evidence"] = ["NOT ACTUALLY PRINTED"]
    else:
        approval["decisions"][0]["findingSha256"] = "0" * 64
    write(path, approval)
    with pytest.raises(ValueError, match=message):
        publish_campaign(campaign)
    assert not (campaign.output / "manifest.json").exists()


def test_unrequested_candidates_do_not_silently_join_or_hide_outside_publication(campaign):
    write(campaign.output / "candidates" / "unexpected.json", {"documentId": "unexpected"})
    with pytest.raises(ValueError, match="exact configured publication scope"):
        publish_campaign(campaign)


def test_validation_only_does_not_publish_and_existing_different_publication_is_preserved(campaign):
    result = publish_campaign(campaign, publish=False)
    assert not result["published"] and result["valid"] == 4
    assert not (campaign.output / "dataset.jsonl").exists()
    path = campaign.output / "manifest.json"
    path.write_text("previous publication")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        publish_campaign(campaign)
    assert path.read_text() == "previous publication"
    assert not (campaign.output / "dataset.jsonl").exists()


def test_source_change_is_not_mislabeled_as_validated_campaign(campaign):
    path = campaign.root / "source" / "train.jsonl"
    path.write_text(path.read_text().replace("SOURCE source_a", "CHANGED source_a"))
    with pytest.raises(ValueError, match="current source differs"):
        publish_campaign(campaign)


def test_conflicting_interrupted_bundle_is_detected_before_any_publication(campaign):
    (campaign.output / "manifest.json.pending").write_text("different interrupted run")
    with pytest.raises(ValueError, match="different interrupted publication"):
        publish_campaign(campaign)
    assert not (campaign.output / "dataset.jsonl").exists()
    assert not (campaign.output / "samples.md").exists()
