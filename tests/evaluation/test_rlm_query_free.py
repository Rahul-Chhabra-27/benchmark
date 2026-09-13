# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The query-agnostic selectors, the tag that keeps them apart, and the two guards.

Pure: no client, no torch, no GPU -- in the style of test_rlm_chunkset.py.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.rlm.chunkset import ChunkRecord, ChunkSet, ChunkSetError, document_digest, window_spans
from evaluation.rlm.query_free_chunks import every_window, lead, random_windows, stride, tag_for
from evaluation.rlm.run_benchmark import build_run_dir_components
from tests.evaluation.test_rlm_run_benchmark import _args

DOCUMENT = ("alpha beta gamma delta epsilon zeta eta theta " * 400)[:9000]
GRID = window_spans(len(DOCUMENT), 2000, 400)  # 6 windows


class SelectorTest(unittest.TestCase):
    """Pure functions of (n_windows, k, seed) -- no document, no question, no gold."""

    def test_lead_takes_the_front_of_the_document(self):
        self.assertEqual(lead(298, 5), [0, 1, 2, 3, 4])

    def test_lead_cannot_run_off_the_end_of_a_short_grid(self):
        self.assertEqual(lead(3, 5), [0, 1, 2])

    def test_stride_spans_the_document_endpoint_to_endpoint(self):
        chosen = stride(298, 5)
        self.assertEqual(chosen[0], 0)
        self.assertEqual(chosen[-1], 297)
        self.assertEqual(len(chosen), 5)

    def test_stride_of_one_takes_the_middle_rather_than_an_endpoint(self):
        self.assertEqual(stride(298, 1), [149])

    def test_stride_returns_fewer_windows_rather_than_a_duplicate(self):
        # Two evenly-spaced indices can round to the same window on a small grid.
        # _cluster_hits would merge the duplicate while the payload cap still counted
        # it twice, so k-1 distinct windows is the honest answer.
        chosen = stride(5, 4)
        self.assertEqual(len(chosen), len(set(chosen)))

    def test_random_is_reproducible_from_the_same_seed(self):
        self.assertEqual(random_windows(298, 5, "0:ex"), random_windows(298, 5, "0:ex"))

    def test_random_draws_independently_per_row_and_per_seed(self):
        # One draw shared by every example would measure that draw. An independent
        # draw per row makes the run mean an estimator of the expected score.
        self.assertNotEqual(random_windows(298, 5, "0:row-a"), random_windows(298, 5, "0:row-b"))
        self.assertNotEqual(random_windows(298, 5, "0:row-a"), random_windows(298, 5, "1:row-a"))

    def test_random_never_repeats_a_window(self):
        chosen = random_windows(298, 10, "0:ex")
        self.assertEqual(len(chosen), len(set(chosen)))

    def test_exhaustive_covers_the_document_with_no_gap(self):
        # Not concatenation: the grid strides by chunk_chars - overlap, so consecutive
        # windows deliberately share `overlap` characters. The union is what must tile.
        spans = [GRID[w] for w in every_window(len(GRID))]
        covered = set()
        for start, end in spans:
            covered |= set(range(start, end))
        self.assertEqual(covered, set(range(len(DOCUMENT))))

    def test_seeded_arms_carry_the_seed_in_the_tag(self):
        # random-s0 and random-s1 are different arms and must not share a directory.
        self.assertEqual(tag_for("random", 0), "random-s0")
        self.assertNotEqual(tag_for("random", 0), tag_for("random", 1))
        self.assertEqual(tag_for("stride", 7), "stride")


class SelectorTagTest(unittest.TestCase):
    def _set(self, *tags, windows=(0, 2)):
        return ChunkSet(
            [
                ChunkRecord(
                    id=f"ex-{i}",
                    windows=tuple(windows),
                    document_sha256=document_digest(DOCUMENT),
                    document_chars=len(DOCUMENT),
                    selector_tag=tag,
                )
                for i, tag in enumerate(tags)
            ],
            chunk_chars=2000,
            overlap=400,
        )

    def test_a_file_with_no_tag_reports_none(self):
        # Every chunk file written before these selectors existed -- the oracle sets
        # and the two hand-made ones -- has no tag, and must keep its run directory.
        self.assertEqual(self._set("", "").tag, "")

    def test_one_tag_is_the_files_tag(self):
        self.assertEqual(self._set("stride", "stride").tag, "stride")

    def test_a_file_mixing_two_selectors_is_refused(self):
        with self.assertRaises(ChunkSetError):
            self._set("lead", "stride").tag

    def test_a_tag_that_would_fake_a_directory_component_is_refused(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.jsonl"
            for bad in ("has/slash", "has__double", "HasUpper", "x" * 30):
                path.write_text(
                    json.dumps(
                        {
                            "id": "ex-1",
                            "windows": [0],
                            "document_sha256": document_digest(DOCUMENT),
                            "document_chars": len(DOCUMENT),
                            "selector_tag": bad,
                        }
                    )
                    + "\n"
                )
                with self.assertRaises(ChunkSetError, msg=f"{bad!r} should be refused"):
                    ChunkSet.from_file(path, chunk_chars=2000, overlap=400)


class MaxClusterCharsTest(unittest.TestCase):
    """What `_cluster_hits` would merge, computed before the run rather than after."""

    def _set(self, windows):
        return ChunkSet(
            [
                ChunkRecord(
                    id="ex-1",
                    windows=tuple(windows),
                    document_sha256=document_digest(DOCUMENT),
                    document_chars=len(DOCUMENT),
                )
            ],
            chunk_chars=2000,
            overlap=400,
        )

    def test_an_exhaustive_selection_collapses_to_the_whole_document(self):
        # The grid overlaps by construction, so every window touches its neighbour and
        # the map step would send ONE passage spanning the corpus.
        self.assertEqual(self._set(every_window(len(GRID))).max_cluster_chars(), len(DOCUMENT))

    def test_a_spread_selection_stays_one_window_wide(self):
        self.assertEqual(self._set([0, 3]).max_cluster_chars(), 2000)

    def test_it_agrees_with_the_clustering_the_map_step_actually_does(self):
        from evaluation.rlm.rlm import _cluster_hits
        from evaluation.rlm.retrieval import Hit

        for windows in ([0, 1, 2], [0, 3], [0, 1, 4, 5], every_window(len(GRID))):
            hits = [
                Hit(rank=r, start=GRID[w][0], end=GRID[w][1], score=0.0, text=DOCUMENT[GRID[w][0] : GRID[w][1]])
                for r, w in enumerate(windows, start=1)
            ]
            widest = max(max(h.end for h in c) - min(h.start for h in c) for c in _cluster_hits(hits))
            self.assertEqual(self._set(windows).max_cluster_chars(), widest, f"windows={windows}")

    def test_max_windows_is_what_a_call_budget_has_to_clear(self):
        self.assertEqual(self._set(every_window(len(GRID))).max_windows, len(GRID))


class RunDirTagTest(unittest.TestCase):
    def test_an_untagged_file_arm_keeps_the_name_it_already_had(self):
        # The invariant the whole mechanism rests on: the four oracle arms on the NAS
        # are `chunks-file5`, and a rename would orphan their checkpoints.
        args = _args(mode="structured", chunks_source="file", chunks_file="x.jsonl", chunks_k=5)
        self.assertNotIn("sel-", "__".join(build_run_dir_components(args, "structured", None)))

    def test_each_selector_claims_its_own_directory(self):
        names = set()
        for tag in ("lead", "stride", "random-s0", "random-s1", "all"):
            args = _args(mode="structured", chunks_source="file", chunks_file="x.jsonl", chunks_k=5)
            args.chunks_tag = tag
            name = "__".join(build_run_dir_components(args, "structured", None))
            self.assertIn(f"sel-{tag}", name)
            names.add(name)
        self.assertEqual(len(names), 5, "two selectors would share one run directory and one checkpoint")

    def test_the_agentic_provided_chunk_arm_gets_the_tag_too(self):
        args = _args(mode="rlm", search_k=5, chunks_source="file", chunks_file="x.jsonl", chunks_k=5)
        args.chunks_tag = "stride"
        self.assertIn("sel-stride", "__".join(build_run_dir_components(args, "rlm", None)))


class GuardTest(unittest.TestCase):
    """Two ways an exhaustive arm reports a sweep it never ran. Both refused up front."""

    def _run(self, *argv):
        import contextlib
        import io
        import sys
        from unittest import mock

        import evaluation.rlm.run_benchmark as rb

        with mock.patch.object(sys, "argv", ["run_benchmark", *argv]), self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                rb.main()
        return err.getvalue()

    def _chunk_file(self, tmp, windows, tag="all"):
        path = Path(tmp) / "chunks.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "loft:nq_128k:0",
                    "windows": list(windows),
                    "spans": [list(GRID[w]) for w in windows],
                    "document_sha256": document_digest(DOCUMENT),
                    "document_chars": len(DOCUMENT),
                    "selector_tag": tag,
                }
            )
            + "\n"
        )
        return str(path)

    BASE = ("--dataset", "loft", "--data-dir", "nq_128k", "--mode", "structured", "--chunks-source", "file")

    def test_an_exhaustive_set_at_passage_granularity_is_refused(self):
        # Every window touches its neighbour, so `passage` sends ONE call spanning the
        # corpus and _cap_subcall truncates it -- a full sweep reported, a prefix read.
        with TemporaryDirectory() as tmp:
            message = self._run(
                *self.BASE,
                "--chunks-file", self._chunk_file(tmp, every_window(len(GRID))),
                "--chunks-k", "6",
                "--max-subcall-chars", "4000",
                "--max-sub-calls", "0",
            )
        self.assertIn("--structured-map-granularity window", message)
        self.assertIn("single 9000-char passage", message)

    def test_a_spread_selection_at_passage_granularity_is_not_refused(self):
        with TemporaryDirectory() as tmp:
            message = self._run(
                *self.BASE,
                "--chunks-file", self._chunk_file(tmp, [0, 3]),
                "--chunks-k", "6",
                "--max-subcall-chars", "4000",
                "--limit", "0",
            )
        self.assertNotIn("--structured-map-granularity window", message)
        self.assertIn("--limit must be positive", message)  # it stopped for an unrelated reason

    def test_a_sweep_wider_than_the_sub_call_budget_is_refused(self):
        # Past the budget llm_query returns a NOTICE STRING, and the findings filter
        # counts it as a finding -- so a truncated sweep reports MORE findings than a
        # healthy one and passes every gate that looks for a starved cell.
        with TemporaryDirectory() as tmp:
            message = self._run(
                *self.BASE,
                "--chunks-file", self._chunk_file(tmp, every_window(len(GRID))),
                "--chunks-k", "6",
                "--structured-map-granularity", "window",
                "--max-sub-calls", "4",
            )
        self.assertIn("--max-sub-calls 0", message)
        self.assertIn("counts as a FINDING", message)

    def test_disabling_the_budget_satisfies_the_guard(self):
        with TemporaryDirectory() as tmp:
            message = self._run(
                *self.BASE,
                "--chunks-file", self._chunk_file(tmp, every_window(len(GRID))),
                "--chunks-k", "6",
                "--structured-map-granularity", "window",
                "--max-sub-calls", "0",
                "--limit", "0",
            )
        self.assertNotIn("counts as a FINDING", message)
        self.assertIn("--limit must be positive", message)

    def test_a_tag_disagreeing_with_the_file_is_refused(self):
        with TemporaryDirectory() as tmp:
            message = self._run(
                *self.BASE,
                "--chunks-file", self._chunk_file(tmp, [0, 3], tag="stride"),
                "--chunks-tag", "lead",
                "--chunks-k", "6",
            )
        self.assertIn("disagrees with the chunk file", message)


if __name__ == "__main__":
    unittest.main()
