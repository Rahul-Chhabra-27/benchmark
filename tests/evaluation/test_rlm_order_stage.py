# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rlm_order_stage: it stages exactly the tasks the campaign reads, and refuses to stage offline."""

import sys
from unittest import mock

import pytest

from evaluation.benchmarks.registry import RULER_32K_TASKS
from evaluation.rlm import rlm_order_stage
from evaluation.rlm.rlm_order_stage import tasks_for


def test_the_campaign_tasks_are_staged():
    loft = tasks_for("loft")
    assert loft == [("loft", t) for t in ("nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k")]
    assert tasks_for("ruler128k") == [("ruler128k", t) for t in RULER_32K_TASKS]
    assert tasks_for("all") == loft + tasks_for("ruler128k")


def test_staging_refuses_the_offline_flags(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with mock.patch.object(sys, "argv", ["rlm_order_stage", "--dataset", "loft"]):
        with pytest.raises(SystemExit):
            rlm_order_stage.main()
