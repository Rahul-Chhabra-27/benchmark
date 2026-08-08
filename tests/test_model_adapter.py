from types import SimpleNamespace

import torch

from kvpress.model_adapter import (
    Qwen35ModelAdapter,
    StandardModelAdapter,
    get_model_adapter,
)


def _standard_model():
    layers = [SimpleNamespace(self_attn=object()), SimpleNamespace(self_attn=object())]
    language_model = SimpleNamespace(layers=layers)
    return SimpleNamespace(
        config=SimpleNamespace(
            model_type="qwen3",
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            hidden_size=512,
            head_dim=64,
        ),
        model=language_model,
        dtype=torch.float16,
    )


def _qwen35_model():
    layers = [
        SimpleNamespace(self_attn="full-0"),
        SimpleNamespace(linear_attn="linear-1"),
        SimpleNamespace(self_attn="full-2"),
    ]
    text_config = SimpleNamespace(
        layer_types=["full_attention", "linear_attention", "full_attention"],
        num_key_value_heads=2,
        head_dim=64,
    )
    return SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5", text_config=text_config),
        model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)),
        dtype=torch.float16,
    )


def test_qwen3_uses_standard_adapter():
    model = _standard_model()
    adapter = get_model_adapter(model)
    assert isinstance(adapter, StandardModelAdapter)
    assert [idx for idx, _ in adapter.iter_kv_attention_layers(model)] == [0, 1]
    assert adapter.kv_bytes_per_token(model) == 2 * 2 * 2 * 64 * 2


def test_qwen35_only_exposes_full_attention_layers():
    model = _qwen35_model()
    adapter = get_model_adapter(model)
    assert isinstance(adapter, Qwen35ModelAdapter)
    assert list(adapter.iter_kv_attention_layers(model)) == [(0, "full-0"), (2, "full-2")]
    assert adapter.kv_bytes_per_token(model) == 2 * 2 * 2 * 64 * 2


def test_qwen35_cache_snapshot_restores_attention_and_recurrent_state():
    adapter = Qwen35ModelAdapter()
    cache = SimpleNamespace(
        key_cache=[torch.ones(1), None],
        value_cache=[torch.ones(1) * 2, None],
        recurrent_states=[torch.ones(2), torch.ones(3)],
        conv_states=[torch.ones(4), torch.ones(5)],
        layer_types=["full_attention", "linear_attention"],
        transformer_layers=[0],
        last_linear_layer=1,
    )
    snapshot = adapter.snapshot_cache_state(cache)
    cache.key_cache[0] = torch.zeros(1)
    cache.value_cache[0] = torch.zeros(1)
    cache.recurrent_states[1] = torch.zeros(3)
    cache.conv_states[1] = torch.zeros(5)
    adapter.restore_cache_state(cache, snapshot)
    assert torch.equal(cache.key_cache[0], torch.ones(1))
    assert torch.equal(cache.value_cache[0], torch.ones(1) * 2)
    assert torch.equal(cache.recurrent_states[1], torch.ones(3))
    assert torch.equal(cache.conv_states[1], torch.ones(5))
