# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The chunk-geometry axes: window/overlap naming, read exposure, and the second hop.

Pure: fake clients, no torch, no GPU, no network.

The load-bearing test here is `test_every_window_cell_gets_its_own_directory`. Before
this axis existed the window slug was gated on `mode == "rlm" and args.search_k`, and
`--mode structured` rejects `--search-k` -- so every cell of a window sweep named the
SAME directory. The first cell ran and every later one aborted on a resume conflict.
The sweep was not merely mis-measured; it could not be run.
"""

import unittest

from evaluation.compare import headline_score, loft_score_key
from evaluation.rlm.chunkset import ChunkSet, ChunkSetError, document_digest, window_spans
from evaluation.rlm.oracle_chunks import gold_windows, select
from evaluation.rlm.retrieval import Hit
from evaluation.rlm.rlm import RLM, DEFAULT_REDUCE_FINDING_CHARS, _hop_query, _truncate_hits, bare_query
from evaluation.rlm.run_benchmark import answer_list_example, build_run_dir_components

from tests.evaluation.test_rlm_run_benchmark import _args

DOCUMENT = "".join(f"passage {i} mentions widget {i} and nothing else at all. " for i in range(400))


def hits(*windows):
    return [Hit(rank=r, start=s, end=e, score=1.0, text=DOCUMENT[s:e]) for r, (s, e) in enumerate(windows, start=1)]


class _Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.asked: list = []
        self.calls = 0
        self.usage = None

    def chat(self, messages, **kw):
        self.asked.append(messages[-1]["content"])
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def build(sub_replies, root_replies=("Final Answer: whale",), **kw):
    root = _Client(root_replies)
    sub = _Client(sub_replies)
    return root, sub, RLM(root_client=root, sub_client=sub, **kw)


class RunDirGeometryTest(unittest.TestCase):
    def _structured(self, **kw):
        args = _args(mode="structured", chunks_source="bm25", **kw)
        return build_run_dir_components(args, "structured", None)

    def test_a_default_structured_run_keeps_the_name_it_already_had(self):
        # The whole campaign on the NAS is named this way. A new axis that renamed
        # it would orphan every checkpoint and silently restart four finished arms.
        self.assertEqual(
            self._structured(split="test"),
            ["loft", "nq_128k", "Qwen_Qwen3-4B-Instruct-2507", "structured", "split-test", "chunks-bm255"],
        )

    def test_every_window_cell_gets_its_own_directory(self):
        names = {
            "__".join(self._structured(search_window=w, search_overlap=w // 5, chunks_k=k))
            for w in (1000, 2000, 4000, 8000)
            for k in (2, 5, 10)
        }
        self.assertEqual(len(names), 12)

    def test_the_window_component_only_appears_when_it_is_not_the_default(self):
        self.assertNotIn("win2000o400", "__".join(self._structured()))
        self.assertIn("win4000o800", "__".join(self._structured(search_window=4000, search_overlap=800)))

    def test_exposure_finding_cap_and_hops_each_split_the_directory(self):
        base = "__".join(self._structured())
        for kw in (
            {"structured_map_chars": 800},
            {"reduce_finding_chars": 8000},
            {"structured_hops": 2},
        ):
            with self.subTest(**kw):
                self.assertNotEqual(base, "__".join(self._structured(**kw)))


class ExposureTest(unittest.TestCase):
    def test_truncation_charges_a_merged_cluster_once(self):
        # Cluster members overlap by construction, so a per-hit budget would spend
        # the shared characters twice and hand the reader less than it asked for.
        kept = _truncate_hits(hits((0, 100), (80, 200)), 150, DOCUMENT)
        self.assertEqual([(h.start, h.end) for h in kept], [(0, 150)])

    def test_hits_stay_verbatim_slices_after_truncation(self):
        for hit in _truncate_hits(hits((10, 400)), 90, DOCUMENT):
            self.assertEqual(hit.text, DOCUMENT[hit.start : hit.end])

    def test_map_chars_bounds_what_the_sub_model_is_shown(self):
        chunks = hits((0, 2000), (4000, 6000))
        _, sub, rlm = build(["fact"], structured_map_chars=300, structured_map_granularity="window")
        rlm.run_structured(DOCUMENT, "what?", chunks)
        self.assertEqual(sub.calls, 2)
        # Each call sees the FIRST 300 chars of its own window and no more, so the
        # read size stops being welded to the retrieval window's width.
        self.assertIn(DOCUMENT[0:300], sub.asked[0])
        self.assertNotIn(DOCUMENT[0:301], sub.asked[0])
        self.assertIn(DOCUMENT[4000:4300], sub.asked[1])
        self.assertNotIn(DOCUMENT[4000:4301], sub.asked[1])

    def test_zero_is_off_and_shows_the_whole_passage(self):
        chunks = hits((0, 2000))
        _, sub, rlm = build(["fact"])
        rlm.run_structured(DOCUMENT, "what?", chunks)
        self.assertIn(DOCUMENT[0:2000], sub.asked[0])

    def test_the_reduce_cap_is_a_knob_and_still_defaults_to_2000(self):
        long_finding = "z" * 5000
        root, _, rlm = build([long_finding])
        rlm.run_structured(DOCUMENT, "what?", hits((0, 100)))
        self.assertEqual(len(root.asked[0].split("[1] ")[1].split("\n\n")[0]), DEFAULT_REDUCE_FINDING_CHARS)

        root, _, rlm = build([long_finding], reduce_finding_chars=4000)
        rlm.run_structured(DOCUMENT, "what?", hits((0, 100)))
        self.assertEqual(len(root.asked[0].split("[1] ")[1].split("\n\n")[0]), 4000)

    def test_a_nonsense_setting_is_refused_rather_than_silently_defaulted(self):
        for kw in ({"structured_map_chars": -1}, {"reduce_finding_chars": 0}, {"structured_hops": 3}):
            with self.subTest(**kw), self.assertRaises(ValueError):
                build(["x"], **kw)


class SecondHopTest(unittest.TestCase):
    def test_the_follow_up_query_is_built_from_findings_not_from_windows(self):
        # Blind expansion over retrieved TEXT was measured on this corpus and made
        # recall worse everywhere. Only terms the reader surfaced may be borrowed,
        # and terms already in the question are dropped -- re-weighting those just
        # returns round one.
        query = _hop_query("who directed it", ["The director is Kurosawa", "Kurosawa also wrote it"])
        self.assertIn("kurosawa", query.lower())
        self.assertTrue(query.startswith("who directed it"))
        self.assertEqual(query.lower().count("directed"), 1)

    def test_hop_two_reads_fresh_windows_and_never_re_reads_round_one(self):
        first = hits((0, 200), (400, 600))
        served = hits((0, 200), (1000, 1200), (2000, 2200))
        _, sub, rlm = build(["found kurosawa"], structured_hops=2, structured_map_granularity="window")

        result = rlm.run_structured(DOCUMENT, "who?", first, retrieve=lambda q, k: served)

        self.assertEqual(result.metrics["hop2_chunks"], 2)
        self.assertEqual(result.metrics["map_calls"], 4)
        self.assertEqual(result.metrics["chunks_provided"], 4)

    def test_hop_two_is_off_by_default_and_needs_a_retriever(self):
        for kw, retrieve in (({}, lambda q, k: hits((900, 999))), ({"structured_hops": 2}, None)):
            with self.subTest(**kw):
                _, sub, rlm = build(["found it"], **kw)
                rlm.run_structured(DOCUMENT, "who?", hits((0, 200)), retrieve=retrieve)
                self.assertEqual(sub.calls, 1)

    def test_a_retriever_that_raises_is_recorded_and_does_not_void_the_round(self):
        def boom(query, k):
            raise RuntimeError("index gone")

        _, _, rlm = build(["found it"], structured_hops=2)
        result = rlm.run_structured(DOCUMENT, "who?", hits((0, 200)), retrieve=boom)
        self.assertIn("RuntimeError", result.metrics["hop2_error"])
        self.assertEqual(result.end_reason, "structured_complete")

    def test_nothing_found_in_round_one_means_no_second_round(self):
        # A hop seeded by no findings is exactly the blind expansion this design
        # refuses to do.
        _, sub, rlm = build(["NOTHING RELEVANT"], structured_hops=2)
        rlm.run_structured(DOCUMENT, "who?", hits((0, 200)), retrieve=lambda q, k: hits((900, 999)))
        self.assertEqual(sub.calls, 1)


class OracleChunkTest(unittest.TestCase):
    DOC = ("a" * 500) + " zebra " + ("b" * 500) + " kiwi " + ("c" * 500) + " zebra " + ("d" * 400)

    def test_it_prefers_windows_that_cover_DIFFERENT_answers(self):
        # The point on a multi-value subset: k windows all carrying the same answer
        # are a far worse oracle than k carrying k different ones.
        spans = window_spans(len(self.DOC), 600, 100)
        carried = gold_windows(self.DOC, ["zebra", "kiwi"], spans)
        covered = set().union(*(carried[w] for w in select(carried, 2)))
        self.assertEqual(covered, {0, 1})

    def test_selection_is_deterministic(self):
        spans = window_spans(len(self.DOC), 600, 100)
        carried = gold_windows(self.DOC, ["zebra", "kiwi"], spans)
        self.assertEqual(select(carried, 3), select(carried, 3))

    def test_a_chunk_file_carrying_spans_refuses_a_different_geometry(self):
        # The failure this closes: window ids re-resolved on another --search-window
        # stay in range and resolve to REAL text, so without spans the run reports a
        # plausible score for windows it never chose.
        grid = window_spans(len(self.DOC), 600, 100)
        record = {
            "id": "x",
            "windows": [1],
            "spans": [list(grid[1])],
            "document_sha256": document_digest(self.DOC),
            "document_chars": len(self.DOC),
        }
        import json
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps(record) + "\n")
            path = fh.name

        self.assertEqual(len(ChunkSet.from_file(path, chunk_chars=600, overlap=100).hits_for("x", self.DOC)), 1)
        with self.assertRaises(ChunkSetError):
            ChunkSet.from_file(path, chunk_chars=1200, overlap=200).hits_for("x", self.DOC)


class AnswerListShapeTest(unittest.TestCase):
    """An arm must be asked for the shape it is scored on.

    LOFT ships one bare `Final Answer: ` prefix for every subset, but qampari and
    quest are parsed by a splitter that only accepts a bracketed list. Measured on
    the oracle arm before this existed: three replies naming every gold answer,
    scored `coverage` 0.000.
    """

    def test_only_the_multi_value_loft_subsets_are_shown_a_list(self):
        self.assertTrue(answer_list_example("loft", "qampari_128k", "structured"))
        self.assertTrue(answer_list_example("loft", "quest_128k", "structured"))
        self.assertEqual(answer_list_example("loft", "nq_128k", "structured"), "")
        self.assertEqual(answer_list_example("loft", "hotpotqa_128k", "structured"), "")
        self.assertEqual(answer_list_example("longbench128k", "hotpotqa", "structured"), "")

    def test_the_agentic_and_vanilla_arms_are_left_alone(self):
        # Their numbers are the reference line this campaign is read against;
        # changing their prompt would silently move it.
        for mode in ("rlm", "vanilla", "both"):
            with self.subTest(mode=mode):
                self.assertEqual(answer_list_example("loft", "qampari_128k", mode), "")

    def test_the_note_reaches_the_reduce_prompt_and_only_when_asked(self):
        root, _, rlm = build(["fact"], structured_answer_list='["a", "b"]')
        rlm.run_structured(DOCUMENT, "which ones?", hits((0, 100)), answer_format="Final Answer: ")
        self.assertIn("bracketed", root.asked[0])
        self.assertIn('["a", "b"]', root.asked[0])

        root, _, rlm = build(["fact"])
        rlm.run_structured(DOCUMENT, "which ones?", hits((0, 100)), answer_format="Final Answer: ")
        self.assertNotIn("bracketed", root.asked[0])

    def test_showing_the_list_shape_claims_its_own_directory(self):
        plain = build_run_dir_components(
            _args(mode="structured", chunks_source="bm25", data_dir="qampari_128k", answer_list_format="off"),
            "structured",
            None,
        )
        corrected = build_run_dir_components(
            _args(mode="structured", chunks_source="bm25", data_dir="qampari_128k"), "structured", None
        )
        self.assertNotIn("listfmt", plain)
        self.assertIn("listfmt", corrected)
        # nq is not scored that way, so its name cannot move.
        nq = build_run_dir_components(_args(mode="structured", chunks_source="bm25"), "structured", None)
        self.assertNotIn("listfmt", nq)


class ChunksQueryTest(unittest.TestCase):
    """BM25 must be asked the question, not the page of instructions around it.

    Measured offline over 55 rows/subset, recall of the gold answer string with the
    full LOFT task versus the query inside it: nq r@2 0.00 -> 0.95, r@5 0.65 -> 0.96;
    qampari r@2 0.00 -> 0.95; musique r@5 0.29 -> 0.58. At depth 2 the boilerplate
    outranks the question completely.
    """

    LOFT_TASK = (
        "You will be given a list of documents.\n"
        "Print out the TITLE and ID of the documents you use to answer.\n"
        "Then format the answers into a list.\n"
        "query: which competition did Lloyd Harris win\n"
    )

    def test_the_wrapper_is_stripped_down_to_the_query(self):
        self.assertEqual(bare_query(self.LOFT_TASK), "which competition did Lloyd Harris win")

    def test_a_task_that_is_already_a_question_is_untouched(self):
        # The conservative half: benchmarks whose task IS the question must not move.
        self.assertEqual(bare_query("Who directed Rashomon?"), "Who directed Rashomon?")

    def test_the_corrected_query_claims_its_own_directory(self):
        # The default stays `task` so a no-flag run still reproduces the arms on the
        # NAS byte for byte -- the invariant every --structured-* flag already keeps.
        # The correction is opt-in and NAMED, which also makes it a labelled A/B
        # rather than a silent replacement of the published numbers.
        as_published = build_run_dir_components(_args(mode="structured", chunks_source="bm25"), "structured", None)
        corrected = build_run_dir_components(
            _args(mode="structured", chunks_source="bm25", chunks_query="bare"), "structured", None
        )
        self.assertNotIn("qbare", as_published)
        self.assertIn("qbare", corrected)

    def test_a_provided_chunk_set_is_not_stamped(self):
        # `file` chunks are chosen offline; no BM25 query is asked at all, so the
        # component would name a knob the run never used.
        provided = build_run_dir_components(
            _args(mode="structured", chunks_source="file", chunks_query="bare"), "structured", None
        )
        self.assertNotIn("qbare", provided)


class MultiValueHeadlineTest(unittest.TestCase):
    def test_qampari_and_quest_are_scored_on_coverage_not_a_pinned_metric(self):
        # compute_multi_value_subspan_em demands EVERY gold be matched, which floors
        # these two subsets. qampari's own vanilla arm reads 0.364 on coverage.
        self.assertEqual(loft_score_key("qampari_128k"), "coverage")
        self.assertEqual(loft_score_key("quest_1m"), "coverage")
        self.assertEqual(loft_score_key("nq_128k"), "subspan_em")
        metrics = {"coverage": 0.364, "subspan_em": 0.018, "em": 0.0}
        self.assertEqual(headline_score(metrics, "loft", "qampari_128k"), ("coverage", 0.364))

    def test_omitting_the_task_reproduces_the_previous_behaviour_exactly(self):
        metrics = {"subspan_em": 0.4727, "em": 0.4, "f1": 0.4267}
        self.assertEqual(headline_score(metrics, "loft"), ("subspan_em", 0.4727))
        self.assertEqual(headline_score(metrics, "loft", "nq_128k"), ("subspan_em", 0.4727))


class StructuredPayloadGuardTest(unittest.TestCase):
    """A cap that shrinks a cell must refuse, not silently measure a smaller one.

    The --search-k arms already refuse this; --mode structured reached none of that
    check, because it rejects --search-k and the guard was gated on it.
    """

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

    BASE = ("--dataset", "loft", "--data-dir", "nq_128k", "--mode", "structured", "--chunks-source", "bm25")

    def test_a_window_wider_than_the_cap_is_refused_up_front(self):
        message = self._run(*self.BASE, "--search-window", "30000", "--search-overlap", "400")
        self.assertIn("Raise --max-subcall-chars", message)

    def test_at_granularity_all_the_whole_k_by_window_payload_has_to_fit(self):
        message = self._run(
            *self.BASE, "--search-window", "8000", "--chunks-k", "10", "--structured-map-granularity", "all"
        )
        self.assertIn("80000 chars", message)

    def test_a_cell_that_fits_is_not_refused_for_this_reason(self):
        # Same geometry, one call per passage: the binding size is one window.
        message = self._run(*self.BASE, "--search-window", "8000", "--chunks-k", "10", "--limit", "0")
        self.assertNotIn("Raise --max-subcall-chars", message)
        self.assertIn("--limit must be positive", message)  # it stopped for an unrelated reason


if __name__ == "__main__":
    unittest.main()
