# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that a loaded Transformers model is using bitsandbytes INT8 layers."""

from typing import Any


def _load_in_8bit_enabled(quantization_config: Any) -> bool:
    if quantization_config is None:
        return False
    if isinstance(quantization_config, dict):
        return quantization_config.get("load_in_8bit") is True
    return getattr(quantization_config, "load_in_8bit", False) is True


def verify_int8_model(model: Any) -> dict[str, int | bool | str]:
    """Fail unless ``model`` was actually loaded with bitsandbytes INT8 weights."""
    try:
        import bitsandbytes as bnb
    except ImportError as error:
        raise RuntimeError("INT8 verification failed: bitsandbytes is not installed") from error

    int8_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, bnb.nn.Linear8bitLt)
    ]
    if not int8_modules:
        raise RuntimeError(
            "INT8 verification failed: the loaded model contains no "
            "bitsandbytes.nn.Linear8bitLt modules"
        )

    model_flag = getattr(model, "is_loaded_in_8bit", None)
    model_quantization_config = getattr(model.config, "quantization_config", None)
    hf_quantizer = getattr(model, "hf_quantizer", None)
    hf_quantization_config = getattr(hf_quantizer, "quantization_config", None)
    config_enabled = _load_in_8bit_enabled(model_quantization_config) or _load_in_8bit_enabled(
        hf_quantization_config
    )

    if model_flag is not True and not config_enabled:
        raise RuntimeError(
            "INT8 verification failed: INT8 layers exist, but neither the model "
            "flag nor its quantization configuration confirms load_in_8bit=True"
        )

    int8_parameters = sum(module.weight.numel() for _, module in int8_modules)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if int8_parameters <= 0 or total_parameters <= 0:
        raise RuntimeError("INT8 verification failed: invalid model parameter counts")

    return {
        "verified": True,
        "backend": "bitsandbytes",
        "int8_linear_modules": len(int8_modules),
        "int8_weight_parameters": int8_parameters,
        "total_parameters": total_parameters,
    }
