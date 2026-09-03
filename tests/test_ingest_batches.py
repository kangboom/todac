"""Standalone tests: python -m unittest discover -s tests -p test_ingest_batches.py."""
import json
import unittest
import weakref
from dataclasses import dataclass, field

from app.worker.ingest_batches import embed_and_insert_batches


@dataclass
class Chunk:
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class Vector(list):
    pass


class IngestBatchTests(unittest.TestCase):
    def test_preserves_content_and_releases_vectors_before_next_batch(self):
        chunks = [Chunk(f"text-{index}", index) for index in range(5)]
        chunks[0].metadata = {"Header 2": "Details", "Header 1": "Guide", "ignored": "value"}
        calls = []
        vector_refs = []
        saved = []

        def embed(texts, batch_number, total_batches):
            self.assertTrue(all(ref() is None for ref in vector_refs))
            calls.append(("embed", len(texts)))
            self.assertEqual(total_batches, 3)
            if batch_number == 1:
                self.assertEqual(texts[0], "Guide > Details\n\ntext-0")
            vectors = [Vector([text]) for text in texts]
            vector_refs.extend(weakref.ref(vector) for vector in vectors)
            return vectors

        def insert(rows, batch_number, total_batches):
            calls.append(("insert", len(rows)))
            # Simulate storage without retaining the input objects.
            saved.extend((row["index"], row["vector"][0], row["headers"]) for row in rows)

        total = embed_and_insert_batches(
            chunks=chunks,
            batch_size=2,
            embed_documents=embed,
            insert_rows=insert,
            build_row=lambda chunk, vector, headers: {
                "index": chunk.chunk_index, "vector": vector, "headers": headers,
            },
        )

        self.assertEqual(calls, [
            ("embed", 2), ("insert", 2),
            ("embed", 2), ("insert", 2),
            ("embed", 1), ("insert", 1),
        ])
        self.assertEqual([row[0] for row in saved], [0, 1, 2, 3, 4])
        self.assertEqual(json.loads(saved[0][2]), {"Header 2": "Details", "Header 1": "Guide"})
        self.assertEqual(saved[1][1:], ("text-1", "{}"))
        self.assertTrue(all(ref() is None for ref in vector_refs))
        self.assertEqual(total, 5)

    def test_rejects_embedding_count_mismatch_before_insert(self):
        inserts = []
        attempts = []
        with self.assertRaisesRegex(ValueError, "Embedding count mismatch"):
            embed_and_insert_batches(
                chunks=[Chunk("a", 0), Chunk("b", 1)],
                batch_size=2,
                embed_documents=lambda *_: [[0.1]],
                insert_rows=lambda rows, *_: inserts.append(rows),
                build_row=lambda chunk, vector, headers: {},
                before_insert=lambda: attempts.append(True),
            )
        self.assertEqual(inserts, [])
        self.assertEqual(attempts, [])

    def test_marks_write_attempt_before_partial_insert_failure(self):
        for fail_on in (1, 2):
            with self.subTest(fail_on=fail_on):
                attempts = []
                completed = []

                def insert(rows, batch_number, total_batches):
                    self.assertEqual(len(attempts), batch_number)
                    if batch_number == fail_on:
                        raise RuntimeError("Milvus unavailable")

                with self.assertRaisesRegex(RuntimeError, "Milvus unavailable"):
                    embed_and_insert_batches(
                        chunks=[Chunk(str(index), index) for index in range(3)],
                        batch_size=2,
                        embed_documents=lambda texts, *_: [[text] for text in texts],
                        insert_rows=insert,
                        build_row=lambda chunk, vector, headers: {"index": chunk.chunk_index},
                        before_insert=lambda: attempts.append(True),
                        after_insert=lambda batch, total, inserted: completed.append(batch),
                    )
                self.assertEqual(len(attempts), fail_on)
                self.assertEqual(completed, list(range(1, fail_on)))

    def test_later_embedding_failure_preserves_prior_write_marker(self):
        attempts = []
        completed = []

        def embed(texts, batch_number, total_batches):
            if batch_number == 2:
                raise RuntimeError("Embedding unavailable")
            return [[text] for text in texts]

        with self.assertRaisesRegex(RuntimeError, "Embedding unavailable"):
            embed_and_insert_batches(
                chunks=[Chunk(str(index), index) for index in range(3)],
                batch_size=2,
                embed_documents=embed,
                insert_rows=lambda *_: None,
                build_row=lambda chunk, vector, headers: {},
                before_insert=lambda: attempts.append(True),
                after_insert=lambda batch, total, inserted: completed.append(inserted),
            )
        self.assertEqual(attempts, [True])
        self.assertEqual(completed, [2])

    def test_invalid_batch_size_is_rejected_without_side_effects(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            embed_and_insert_batches(
                chunks=[Chunk("a", 0)], batch_size=0,
                embed_documents=None, insert_rows=None, build_row=None,
            )


if __name__ == "__main__":
    unittest.main()
