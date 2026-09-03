"""Bounded-memory embedding and storage batches for document ingestion."""
import json


def embed_and_insert_batches(
    *,
    chunks,
    batch_size,
    embed_documents,
    insert_rows,
    build_row,
    before_insert=None,
    after_insert=None,
):
    """Embed and persist at most batch_size chunks at a time."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    chunk_count = len(chunks)
    total_batches = (chunk_count + batch_size - 1) // batch_size
    total_inserted = 0

    for batch_number, start in enumerate(range(0, chunk_count, batch_size), start=1):
        batch = chunks[start:start + batch_size]
        embedding_texts = []
        headers_json = []

        for chunk in batch:
            header_metadata = {
                key: value
                for key, value in chunk.metadata.items()
                if key.startswith("Header")
            }
            if header_metadata:
                header_path = " > ".join(
                    header_metadata[key] for key in sorted(header_metadata)
                )
                embedding_texts.append(f"{header_path}\n\n{chunk.text}")
            else:
                embedding_texts.append(chunk.text)
            headers_json.append(
                json.dumps(header_metadata, ensure_ascii=False)
                if header_metadata
                else "{}"
            )

        vectors = embed_documents(embedding_texts, batch_number, total_batches)
        if len(vectors) != len(batch):
            raise ValueError(
                f"Embedding count mismatch: expected {len(batch)}, got {len(vectors)}"
            )

        rows = [
            build_row(chunk, vector, header_json)
            for chunk, vector, header_json in zip(batch, vectors, headers_json)
        ]

        if before_insert is not None:
            before_insert()
        insert_rows(rows, batch_number, total_batches)

        total_inserted += len(rows)
        if after_insert is not None:
            after_insert(batch_number, total_batches, total_inserted)

        # Drop the previous vectors before calling the embedding API again.
        # Reassigning rows next iteration would keep two batches alive at once.
        del batch, embedding_texts, headers_json, vectors, rows

    return total_inserted
