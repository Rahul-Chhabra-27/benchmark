"""Small model-specific shims used by KVPress.

The standard adapter intentionally mirrors the cache and layer handling that
KVPress used before hybrid Qwen3.5 models were added.  Qwen3.5 has a mixture
of full-attention and recurrent layers, so only its full-attention layers are
presented to KVPress.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch
from transformers import DynamicCache, QuantizedCache


class ModelAdapter:
    def get_text_config(self, model):
        raise NotImplementedError

    def get_language_model(self, model):
        raise NotImplementedError

    def iter_kv_attention_layers(self, model):
        raise NotImplementedError

    def create_cache(self, model):
        raise NotImplementedError

    def kv_bytes_per_token(self, model, batch_size=1):
        raise NotImplementedError

    def snapshot_cache_state(self, cache):
        raise NotImplementedError

    def restore_cache_state(self, cache, snapshot):
        raise NotImplementedError

    # These helpers keep cache-specific details out of the press and pipeline.
    def get_keys_and_values(self, cache, layer_idx: int):
        layer = cache.layers[layer_idx]
        if isinstance(cache, QuantizedCache):
            return layer._dequantize(layer._quantized_keys), layer._dequantize(layer._quantized_values)
        return layer.keys, layer.values

    def set_keys_and_values(self, cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
        layer = cache.layers[layer_idx]
        if isinstance(cache, QuantizedCache):
            layer._quantized_keys = layer._quantize(keys, axis=layer.axis_key)
            layer._quantized_values = layer._quantize(values, axis=layer.axis_value)
            layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            layer.values = torch.zeros(0, dtype=values.dtype, device=values.device)
            layer.cumulative_length = keys.shape[-2]
        else:
            layer.keys = keys
            layer.values = values

    def cache_seq_lengths(self, cache) -> list[int]:
        return [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]

    def truncate_cache(self, cache, cache_seq_lengths: list[int]) -> None:
        for layer_idx, sequence_length in enumerate(cache_seq_lengths):
            layer = cache.layers[layer_idx]
            if layer.keys is not None:
                layer.keys = layer.keys[:, :, :sequence_length]
                layer.values = layer.values[:, :, :sequence_length]
            if isinstance(cache, QuantizedCache):
                layer._quantized_keys = layer._quantized_keys[:, :, :sequence_length]
                layer._quantized_values = layer._quantized_values[:, :, :sequence_length]

    @property
    def is_qwen35(self) -> bool:
        return False


class StandardModelAdapter(ModelAdapter):
    def get_text_config(self, model):
        return model.config

    def get_language_model(self, model):
        return model.model.language_model if hasattr(model.model, "language_model") else model.model

    def iter_kv_attention_layers(self, model) -> Iterator[tuple[int, Any]]:
        for layer_idx, layer in enumerate(self.get_language_model(model).layers):
            yield layer_idx, layer.self_attn

    def create_cache(self, model):
        return DynamicCache()

    def kv_bytes_per_token(self, model, batch_size=1):
        config = self.get_text_config(model)
        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        bytes_per_element = torch.finfo(model.dtype).bits // 8
        return num_layers * 2 * num_kv_heads * head_dim * bytes_per_element * batch_size

    def snapshot_cache_state(self, cache):
        layers = []
        for layer in cache.layers:
            state = {name: getattr(layer, name) for name in ("keys", "values") if hasattr(layer, name)}
            for name in ("_quantized_keys", "_quantized_values", "cumulative_length"):
                if hasattr(layer, name):
                    state[name] = getattr(layer, name)
            layers.append(state)
        metadata = {
            name: getattr(cache, name)
            for name in ("_seen_tokens", "seen_tokens", "cache_position")
            if hasattr(cache, name)
        }
        return {"layers": layers, "metadata": metadata}

    def restore_cache_state(self, cache, snapshot):
        for layer, state in zip(cache.layers, snapshot["layers"]):
            for name, value in state.items():
                setattr(layer, name, value)
        for name, value in snapshot["metadata"].items():
            setattr(cache, name, value)


class Qwen35ModelAdapter(ModelAdapter):
    @property
    def is_qwen35(self) -> bool:
        return True

    def get_text_config(self, model):
        return model.config.text_config

    def get_language_model(self, model):
        return model.model.language_model

    def iter_kv_attention_layers(self, model) -> Iterator[tuple[int, Any]]:
        language_model = self.get_language_model(model)
        text_config = self.get_text_config(model)
        for layer_idx, layer in enumerate(language_model.layers):
            if text_config.layer_types[layer_idx] == "full_attention":
                yield layer_idx, layer.self_attn

    def create_cache(self, model):
        # Recent Transformers releases provide a hybrid cache with the exact
        # recurrent/convolution state fields used by Qwen3.5.  Fall back to
        # the requested DynamicCache construction on releases without it.
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

            return Qwen3_5DynamicCache(config=self.get_text_config(model))
        except (ImportError, TypeError):  # pragma: no cover - older Transformers
            return DynamicCache(config=self.get_text_config(model))

    def kv_bytes_per_token(self, model, batch_size=1):
        text_config = self.get_text_config(model)
        num_kv_layers = text_config.layer_types.count("full_attention")
        bytes_per_element = torch.finfo(model.dtype).bits // 8
        return (
            num_kv_layers
            * 2
            * text_config.num_key_value_heads
            * text_config.head_dim
            * bytes_per_element
            * batch_size
        )

    def get_keys_and_values(self, cache, layer_idx: int):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]

    def set_keys_and_values(self, cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
        cache.key_cache[layer_idx] = keys
        cache.value_cache[layer_idx] = values

    def cache_seq_lengths(self, cache) -> list[int]:
        return [cache.get_seq_length(layer_idx) for layer_idx in cache.transformer_layers]

    def truncate_cache(self, cache, cache_seq_lengths: list[int]) -> None:
        for layer_idx, sequence_length in zip(cache.transformer_layers, cache_seq_lengths):
            if cache.key_cache[layer_idx] is not None:
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :sequence_length]
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :sequence_length]

    def snapshot_cache_state(self, cache):
        # Qwen3.5 names these fields explicitly; do not infer or alias them.
        return {
            "key_cache": list(cache.key_cache),
            "value_cache": list(cache.value_cache),
            "recurrent_states": list(cache.recurrent_states),
            "conv_states": list(cache.conv_states),
            "metadata": {
                name: getattr(cache, name)
                for name in (
                    "layer_types",
                    "transformer_layers",
                    "last_linear_layer",
                    "_seen_tokens",
                    "seen_tokens",
                    "cache_position",
                )
                if hasattr(cache, name)
            },
        }

    def restore_cache_state(self, cache, snapshot):
        cache.key_cache = list(snapshot["key_cache"])
        cache.value_cache = list(snapshot["value_cache"])
        cache.recurrent_states = list(snapshot["recurrent_states"])
        cache.conv_states = list(snapshot["conv_states"])
        for name, value in snapshot["metadata"].items():
            setattr(cache, name, value)


def get_model_adapter(model) -> ModelAdapter:
    if model.config.model_type == "qwen3_5":
        return Qwen35ModelAdapter()
    return StandardModelAdapter()
