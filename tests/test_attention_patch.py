# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from kvpress.attention_patch import search_hyperplane


def test_search_hyperplane():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bsz, seq_len, head_dim = 50, 500, 128
    X = torch.rand(bsz, seq_len, head_dim, device=device)
    Y = search_hyperplane(X)
    assert torch.exp(torch.bmm(X, Y.unsqueeze(-1))).max() == 0


def test_search_hyperplane_is_finite_for_16_bit_dtypes():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for dtype in (torch.float16, torch.bfloat16):
        X = torch.rand(2, 16, 32, device=device, dtype=dtype)
        Y = search_hyperplane(X)

        assert Y.dtype == dtype
        assert torch.isfinite(Y).all()
        assert Y.abs().max() <= min(1e5, torch.finfo(dtype).max / 8)
        assert torch.exp(torch.bmm(X.float(), Y.float().unsqueeze(-1))).max() == 0
