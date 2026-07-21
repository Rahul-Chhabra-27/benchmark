# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that a loaded Transformers model is using bitsandbytes NF4 layers."""

from typing import Any


def _load_in_4bit_enabled(quantization_config: Any) -> bool:
    if quantization_config is None:
        return False
    if isinstance(quantization_config, dict):
        return quantization_config.get("load_in_4bit") is True
    return getattr(quantization_config, "load_in_4bit", False) is True


def _quant_type(quantization_config: Any) -> str | None:
    if quantization_config is None:
        return None
    if isinstance(quantization_config, dict):
        return quantization_config.get("bnb_4bit_quant_type")
    return getattr(quantization_config, "bnb_4bit_quant_type", None)


def verify_int4_model(model: Any) -> dict[str, int | bool | str]:
    """Fail unless ``model`` was actually loaded with bitsandbytes NF4 weights."""
    try:
        import bitsandbytes as bnb
    except ImportError as error:
        raise RuntimeError("4-bit verification failed: bitsandbytes is not installed") from error

    int4_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, bnb.nn.Linear4bit)
    ]
    if not int4_modules:
        raise RuntimeError(
            "4-bit verification failed: the loaded model contains no "
            "bitsandbytes.nn.Linear4bit modules"
        )

    model_flag = getattr(model, "is_loaded_in_4bit", None)
    model_quantization_config = getattr(model.config, "quantization_config", None)
    hf_quantizer = getattr(model, "hf_quantizer", None)
    hf_quantization_config = getattr(hf_quantizer, "quantization_config", None)
    configs = [model_quantization_config, hf_quantization_config]
    config_enabled = any(_load_in_4bit_enabled(config) for config in configs)
    quant_types = {
        quant_type
        for config in configs
        if config is not None
        for quant_type in [_quant_type(config)]
        if quant_type is not None
    }

    if model_flag is not True and not config_enabled:
        raise RuntimeError(
            "4-bit verification failed: Linear4bit layers exist, but neither the model "
            "flag nor its quantization configuration confirms load_in_4bit=True"
        )
    if "nf4" not in quant_types:
        raise RuntimeError(
            f"4-bit verification failed: expected NF4 quantization, found {sorted(quant_types)}"
        )

    int4_parameters = sum(module.weight.numel() for _, module in int4_modules)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if int4_parameters <= 0 or total_parameters <= 0:
        raise RuntimeError("4-bit verification failed: invalid model parameter counts")

    return {
        "verified": True,
        "backend": "bitsandbytes",
        "quant_type": "nf4",
        "int4_linear_modules": len(int4_modules),
        "int4_weight_parameters": int4_parameters,
        "total_parameters": total_parameters,
    }
