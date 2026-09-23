from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_training_model_contract import _tiny_t5gemma2

from document_ocr.training.config import LoraConfig, load_training_config
from document_ocr.training.eva import activation_indices, initialize_eva, input_streams
from document_ocr.training.runtime import load_lora_model

ROOT = Path(__file__).resolve().parents[1]


def _peft():
    base = load_training_config(ROOT / "configs/training/t5gemma2_270m_lora.pilot106.yaml")
    payload = base.peft.model_dump(mode="python")
    payload.update(
        rank=2,
        alpha=2,
        init_lora_weights="eva",
        eva=dict(
            rho=1.0,
            tau=0.99,
            whiten=False,
            sample_count=4,
            batch_size=1,
            tokens_per_stream=8,
            max_forward_passes=128,
            seed=23,
        ),
    )
    return base, LoraConfig.model_validate(payload)


def test_eva_requires_explicit_and_consistent_configuration():
    _, peft = _peft()
    value = peft.model_dump(mode="python")
    value["eva"] = None
    with pytest.raises(ValidationError, match=r"explicit peft\.eva"):
        LoraConfig.model_validate(value)
    value = peft.model_dump(mode="python")
    value["init_lora_weights"] = True
    with pytest.raises(ValidationError, match="only EVA"):
        LoraConfig.model_validate(value)
    value = peft.model_dump(mode="python")
    value["eva"]["tokens_per_stream"] = 1
    with pytest.raises(ValidationError, match="at least the adapter rank"):
        LoraConfig.model_validate(value)


def test_masks_select_actual_tokens_even_when_sequence_lengths_match():
    import torch

    prefix = "base_model.model.model."
    assert input_streams(prefix + "decoder.layers.0.self_attn.k_proj") == ("decoder", "encoder")
    assert input_streams(prefix + "encoder.text_model.layers.0.self_attn.k_proj") == ("encoder",)
    assert input_streams(prefix + "decoder.layers.0.self_attn.q_proj") == ("decoder",)
    mask = torch.tensor([[1, 0, 1, 1, 0], [0, 1, 0, 0, 1]], dtype=torch.bool)
    indices = activation_indices(mask, 2)
    assert indices.tolist() == [[0, 0], [0, 3], [1, 1], [1, 4]]
    with pytest.raises(ValueError, match="no token-stream contract"):
        input_streams("vision_tower.proj")
    with pytest.raises(ValueError, match="no usable tokens"):
        activation_indices(torch.zeros(1, 5, dtype=torch.bool), 2)


def _fixture(monkeypatch):
    import torch
    from transformers import AutoModelForSeq2SeqLM

    torch.set_num_threads(2)
    torch.manual_seed(9)
    tiny = _tiny_t5gemma2()
    monkeypatch.setattr(AutoModelForSeq2SeqLM, "from_pretrained", lambda *a, **kw: tiny)
    config, peft = _peft()
    config = config.model_copy(
        update={
            "model": config.model.model_copy(update={"token_env": None, "dtype": "float32"}),
            "peft": peft,
        }
    )
    tiny.name_or_path = config.model.name_or_path
    model, _ = load_lora_model(config)
    data = [dict(document_id=f"train-{i}", i=i) for i in range(4)]

    def collate(rows):
        i = rows[0]["i"]
        # Equal source/target lengths deliberately defeat length-based mask guessing.
        labels = torch.tensor([[7 + i, 12, 13, 1, -100, -100]])
        return dict(
            input_ids=torch.tensor([[2, 3 + i, 4, 5, 1, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0]]),
            labels=labels,
            decoder_input_ids=model.prepare_decoder_input_ids_from_labels(labels),
        )

    return model, data, collate, peft


def test_actual_peft_eva_covers_both_shared_kv_streams_and_preserves_base(monkeypatch, tmp_path):
    import torch

    model, data, collate, peft = _fixture(monkeypatch)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n}
    model.eval()
    with torch.no_grad():
        expected = model(**collate(data[:1])).logits.clone()
    # Initialization must not materialize the large vocabulary logits.
    head_handle = model.get_base_model().lm_head.register_forward_hook(
        lambda *args: pytest.fail("EVA called the vocabulary head")
    )
    rng_before = torch.random.get_rng_state().clone()
    try:
        report = initialize_eva(
            model=model,
            train_dataset=data,
            collator=collate,
            peft=peft,
            dataset_identity="test-cache",
            device="cpu",
        )
    finally:
        head_handle.remove()
    assert report["status"] == "complete"
    assert len(report["modules"]) == 14
    assert model.peft_config[peft.adapter_name].target_modules == peft.target_modules_regex
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
    for row in report["modules"]:
        if ".decoder." in row["module"] and row["module"].endswith(("k_proj", "v_proj")):
            assert row["activation_rows"]["encoder"] == report["forward_passes"] * 5
            assert row["activation_rows"]["decoder"] == report["forward_passes"] * 4
    modules = dict(model.named_modules())
    prefix = "base_model.model.model.decoder.layers.0.self_attn."
    bases = {
        key: modules[prefix + key].lora_A[peft.adapter_name].weight.detach()
        for key in ("q_proj", "k_proj", "v_proj")
    }
    # K and V see the same union of streams, whereas Q sees only the decoder.
    # Check subspaces rather than SVD signs; first-call-only deduplication would
    # incorrectly give all three the same initialization subspace.
    projectors = {key: basis.T @ basis for key, basis in bases.items()}
    torch.testing.assert_close(projectors["k_proj"], projectors["v_proj"])
    assert not torch.allclose(projectors["q_proj"], projectors["k_proj"], atol=1e-5)
    with torch.no_grad():
        torch.testing.assert_close(model(**collate(data[:1])).logits, expected)
    for name, parameter in model.named_parameters():
        if name in before:
            torch.testing.assert_close(parameter, before[name])
        elif "lora_A" in name:
            torch.testing.assert_close(parameter @ parameter.T, torch.eye(2), atol=1e-5, rtol=1e-5)
        elif "lora_B" in name:
            assert parameter.count_nonzero() == 0
    assert not any(m._forward_pre_hooks for m in model.modules())
    # Native EVA normally rewrites target_modules. Prove our fixed-rank save
    # contract remains compatible with the existing checkpoint evaluator.
    from tools.evaluate_kie_checkpoint import _validate_adapter_contract

    config, _ = _peft()
    # These fixtures adapt only linear projections; avoid a Hub lookup for an
    # embedding-resize decision that is irrelevant to this save contract.
    model.save_pretrained(tmp_path, save_embedding_layers=False)
    _validate_adapter_contract(
        tmp_path / peft.adapter_name, config.model_copy(update={"peft": peft})
    )


def test_eva_timeout_is_explicit_and_cleans_model_hooks(monkeypatch):
    model, data, collate, peft = _fixture(monkeypatch)
    peft = peft.model_copy(
        update={"eva": peft.eva.model_copy(update={"max_forward_passes": 3, "tau": 1.0})}
    )
    # The actual native config must have the same stopping criterion.
    model.peft_config[peft.adapter_name].eva_config.tau = 1.0
    with pytest.raises(RuntimeError, match="training was not started"):
        initialize_eva(
            model=model,
            train_dataset=data,
            collator=collate,
            peft=peft,
            dataset_identity="test-cache",
            device="cpu",
        )
    assert not any(m._forward_pre_hooks for m in model.modules())


def test_resume_verifies_receipt_without_reinitializing(monkeypatch, tmp_path):
    import document_ocr.training.eva as eva

    _, peft = _peft()
    path = tmp_path / "eva-initialization.json"
    path.write_text(
        json.dumps(
            dict(
                status="complete",
                settings=peft.eva.model_dump(mode="json"),
                dataset_identity="cache-a",
            )
        )
    )
    monkeypatch.setattr(eva, "initialize_eva", lambda **kw: pytest.fail("resume reinitialized EVA"))
    kwargs = dict(
        path=path,
        resuming=True,
        model=None,
        train_dataset=None,
        collator=None,
        peft=peft,
        dataset_identity="cache-a",
        device="cpu",
    )
    eva.initialize_eva_for_run(**kwargs)
    kwargs["dataset_identity"] = "other-cache"
    with pytest.raises(RuntimeError, match="training contract"):
        eva.initialize_eva_for_run(**kwargs)
