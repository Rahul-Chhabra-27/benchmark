#!/usr/bin/env python3
"""Short GPU smoke test for Qwen3.5 + KVzipPress.

This deliberately uses a tiny context. It validates the adapter-aware KVzip
hook/reconstruction path before launching a long-context benchmark.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/rethinkingai-self/25m0820/kvpress/Qwen3.5-9B",
    )
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--compression-ratio", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a GPU compute node.")

    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kvpress import KVzipPress
    from kvpress.model_adapter import get_model_adapter

    print(f"Loading model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.name_or_path = args.model

    seed_text = (
        "This is a short Qwen3.5 KVzip smoke-test context. "
        "The important fact is that the adapter must compress only full-attention layers. "
    )
    seed_ids = tokenizer(seed_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    repeats = (args.context_tokens // max(seed_ids.shape[1], 1)) + 1
    context_ids = seed_ids.repeat(1, repeats)[:, : args.context_tokens].to(model.device)

    adapter = get_model_adapter(model)
    cache = adapter.create_cache(model)
    full_layers = [idx for idx, _ in adapter.iter_kv_attention_layers(model)]
    print(f"Full-attention layers: {full_layers}", flush=True)

    press = KVzipPress(compression_ratio=args.compression_ratio)
    print("Running KVzip prefill and reconstruction...", flush=True)
    with press(model):
        outputs = model.model(
            input_ids=context_ids,
            past_key_values=cache,
            use_cache=True,
        )
    if outputs.past_key_values is not None:
        cache = outputs.past_key_values

    lengths = {
        layer_idx: cache.key_cache[layer_idx].shape[-2]
        for layer_idx in full_layers
        if cache.key_cache[layer_idx] is not None
    }
    print(f"Compressed full-attention cache lengths: {lengths}", flush=True)
    if not lengths or not all(length < args.context_tokens for length in lengths.values()):
        raise AssertionError("KVzip did not shorten the full-attention KV cache")

    snapshot = adapter.snapshot_cache_state(cache)
    question_ids = tokenizer(
        "What is the important fact?",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(model.device)

    with torch.inference_mode():
        first = model(input_ids=question_ids, past_key_values=cache, use_cache=True).logits[:, -1, :]
        first_token = first.argmax(dim=-1)
        adapter.restore_cache_state(cache, snapshot)
        second = model(input_ids=question_ids, past_key_values=cache, use_cache=True).logits[:, -1, :]
        second_token = second.argmax(dim=-1)

    if not torch.equal(first_token, second_token):
        raise AssertionError("Restored cache produced a different first answer token")

    print(f"First generated token after restore: {second_token.tolist()}", flush=True)
    print("Qwen3.5 + KVzip smoke test passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
