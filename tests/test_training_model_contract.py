from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from document_ocr.training.config import load_training_config
from document_ocr.training.runtime import load_lora_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "training" / "t5gemma2_270m_lora.pilot106.yaml"
)


def _tiny_t5gemma2() -> Any:
    from transformers import T5Gemma2Config, T5Gemma2ForConditionalGeneration

    text = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "sliding_window": 32,
        "layer_types": ["full_attention"],
        "pad_token_id": 0,
        "eos_token_id": 1,
        "bos_token_id": 2,
    }
    vision = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_channels": 3,
        "image_size": 8,
        "patch_size": 4,
        "projection_dim": 16,
    }
    encoder = {
        "text_config": text,
        "vision_config": vision,
        "mm_tokens_per_image": 4,
        "boi_token_index": 60,
        "eoi_token_index": 61,
        "image_token_index": 62,
    }
    return T5Gemma2ForConditionalGeneration(
        T5Gemma2Config(encoder=encoder, decoder=text, image_token_index=62)
    )


@pytest.mark.filterwarnings("ignore:Unrecognized keys in `rope_parameters`.*")
def test_real_peft_insertion_targets_only_text_encoder_and_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM

    tiny_model = _tiny_t5gemma2()

    def return_tiny(*args: Any, **kwargs: Any) -> Any:
        return tiny_model

    monkeypatch.setattr(AutoModelForSeq2SeqLM, "from_pretrained", return_tiny)
    config = load_training_config(CONFIG_PATH)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={"token_env": None, "dtype": "float32"}
            )
        }
    )

    model, facts = load_lora_model(config)

    assert facts["adapted_module_count"] == 14
    assert facts["encoder_training_use_cache"] is False
    assert facts["training_use_cache"] is False
    assert facts["training_cache_config_count"] >= 2
    assert facts["generation_use_cache"] is True
    assert model.base_model.model.config.encoder.text_config.use_cache is False
    assert model.base_model.model.config.decoder.use_cache is False
    assert model.base_model.model.model.encoder.text_model.config.use_cache is False
    assert model.base_model.model.model.decoder.config.use_cache is False
    assert model.generation_config.use_cache is True
    assert model.generation_config.top_p is None
    assert model.generation_config.top_k is None
    assert all("vision_tower" not in name for name in facts["adapted_modules"])
    assert all("multi_modal_projector" not in name for name in facts["adapted_modules"])
    assert 0 < facts["trainable_parameters"] < facts["total_parameters"]

    output = model(
        input_ids=torch.tensor([[2, 3, 1]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        labels=torch.tensor([[4, 5, 1]]),
    )
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in trainable
    )
    assert all(parameter.grad is None for parameter in frozen)
