# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pre-selected chunk file, and every way it is allowed to refuse.

Pure: no client, no torch, no GPU -- in the style of test_rlm_retrieval.py.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.rlm.chunk_sources.chunkset import ChunkSet, ChunkSetError, document_digest, file_digest, window_spans
from evaluation.rlm.shared import retrieval

DOCUMENT = ("alpha beta gamma delta epsilon zeta eta theta " * 400)[:9000]


def write(path: Path, records) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def record(example_id="ex-1", windows=(0, 2), document=DOCUMENT, **kw):
    base = {
        "id": example_id,
        "windows": list(windows),
        "document_sha256": document_digest(document),
        "document_chars": len(document),
        "selector": "claude-opus-5",
    }
    base.update(kw)
    return base


class GridTest(unittest.TestCase):
    def test_the_window_grid_matches_the_retrieval_index_exactly(self):
        # The whole point of addressing chunks by window id: a `file` arm and a
        # `bm25` arm must be choosing from one and the same set of windows.
        index = retrieval.ChunkIndex(DOCUMENT, chunk_chars=2000, overlap=400)
        self.assertEqual(window_spans(len(DOCUMENT), 2000, 400), index.spans)

    def test_a_zero_overlap_grid_also_matches(self):
        index = retrieval.ChunkIndex(DOCUMENT, chunk_chars=2000, overlap=0)
        self.assertEqual(window_spans(len(DOCUMENT), 2000, 0), index.spans)

    def test_a_bad_geometry_is_refused(self):
        with self.assertRaises(ValueError):
            window_spans(100, 2000, 2000)


class HitsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def load(self, records, **kw):
        return ChunkSet.from_file(write(self.dir / "chunks.jsonl", records), **kw)

    def test_hits_are_verbatim_slices_of_the_document(self):
        # The invariant the whole span-attribution path rests on: text that is not
        # findable in the document cannot be attributed to it.
        hits = self.load([record(windows=(0, 2))]).hits_for("ex-1", DOCUMENT)
        self.assertEqual(len(hits), 2)
        for hit in hits:
            self.assertEqual(hit.text, DOCUMENT[hit.start : hit.end])
        self.assertEqual([h.rank for h in hits], [1, 2])

    def test_windows_resolve_to_the_same_offsets_the_index_would_give(self):
        hits = self.load([record(windows=(3,))]).hits_for("ex-1", DOCUMENT)
        self.assertEqual((hits[0].start, hits[0].end), window_spans(len(DOCUMENT))[3])

    def test_a_missing_id_is_an_error_not_an_empty_selection(self):
        chunks = self.load([record(example_id="ex-1")])
        with self.assertRaises(ChunkSetError) as caught:
            chunks.hits_for("ex-2", DOCUMENT)
        self.assertIn("ex-2", str(caught.exception))

    def test_a_different_document_of_the_same_length_is_refused(self):
        # The silent-failure case this check exists for: the offsets are all in
        # range, so every example would read real text from the wrong corpus.
        other = "z" * len(DOCUMENT)
        chunks = self.load([record(document=DOCUMENT)])
        with self.assertRaises(ChunkSetError) as caught:
            chunks.hits_for("ex-1", other)
        self.assertIn("sha256", str(caught.exception))

    def test_a_different_length_is_refused_before_the_hash(self):
        chunks = self.load([record(document=DOCUMENT)])
        with self.assertRaises(ChunkSetError) as caught:
            chunks.hits_for("ex-1", DOCUMENT[:-10])
        self.assertIn("char document", str(caught.exception))

    def test_a_window_outside_the_grid_is_refused(self):
        chunks = self.load([record(windows=(999,))])
        with self.assertRaises(ChunkSetError) as caught:
            chunks.hits_for("ex-1", DOCUMENT)
        self.assertIn("outside", str(caught.exception))

    def test_spans_that_disagree_with_windows_are_refused(self):
        chunks = self.load([record(windows=(0,), spans=[[10, 20]])])
        with self.assertRaises(ChunkSetError) as caught:
            chunks.hits_for("ex-1", DOCUMENT)
        self.assertIn("disagree", str(caught.exception))

    def test_matching_spans_are_accepted(self):
        span = list(window_spans(len(DOCUMENT))[1])
        hits = self.load([record(windows=(1,), spans=[span])]).hits_for("ex-1", DOCUMENT)
        self.assertEqual([hits[0].start, hits[0].end], span)

    def test_a_payload_over_the_cap_is_refused(self):
        # Without this an arm handed better chunks would also be handed MORE of
        # them, and the comparison would measure payload size.
        chunks = self.load([record(windows=(0, 1, 2, 3))], max_payload_chars=2 * 2000)
        with self.assertRaises(ChunkSetError) as caught:
            chunks.hits_for("ex-1", DOCUMENT)
        self.assertIn("cap", str(caught.exception))

    def test_a_payload_at_the_cap_is_accepted(self):
        chunks = self.load([record(windows=(0, 1))], max_payload_chars=2 * 2000)
        self.assertEqual(len(chunks.hits_for("ex-1", DOCUMENT)), 2)

    def test_duplicate_ids_are_refused(self):
        with self.assertRaises(ChunkSetError):
            self.load([record(example_id="ex-1"), record(example_id="ex-1")])

    def test_a_record_missing_a_required_field_names_the_line(self):
        bad = record()
        del bad["document_sha256"]
        with self.assertRaises(ChunkSetError) as caught:
            self.load([record(example_id="ok"), bad])
        self.assertIn(":2", str(caught.exception))

    def test_a_missing_file_is_refused(self):
        with self.assertRaises(ChunkSetError):
            ChunkSet.from_file(self.dir / "nope.jsonl")

    def test_selectors_are_reported_for_provenance(self):
        self.assertEqual(self.load([record()]).selectors, ["claude-opus-5"])

    def test_the_file_digest_tracks_content_not_name(self):
        # A path is not identity: the resume guard has to notice an edited file.
        first = write(self.dir / "a.jsonl", [record()])
        same = write(self.dir / "b.jsonl", [record()])
        edited = write(self.dir / "c.jsonl", [record(windows=(1,))])
        self.assertEqual(file_digest(first), file_digest(same))
        self.assertNotEqual(file_digest(first), file_digest(edited))


if __name__ == "__main__":
    unittest.main()
