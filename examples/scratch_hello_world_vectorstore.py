"""
Scratch smoke test for VectorStoreManager (non-production).

What it does:
1) Loads examples/sample_chunk.json
2) Ingests one chunk into a persistent Chroma collection
3) Queries: "what is a neural network"
4) Prints top similarity score (if any)

Run:
    uv run python examples/scratch_hello_world_vectorstore.py
Run twice to verify duplicate skipping on second run.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings
from rag_agent.vectorstore.store import VectorStoreManager


def load_sample_chunk(sample_path: Path) -> DocumentChunk:
    with sample_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    chunk_text = payload["chunk_text"]
    metadata_dict = payload["metadata"]

    metadata = ChunkMetadata(
        topic=metadata_dict["topic"],
        difficulty=metadata_dict["difficulty"],
        type=metadata_dict["type"],
        source=metadata_dict["source"],
        related_topics=metadata_dict.get("related_topics", []),
        is_bonus=metadata_dict.get("is_bonus", False),
    )

    chunk_id = VectorStoreManager.generate_chunk_id(metadata.source, chunk_text)

    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_text=chunk_text,
        metadata=metadata,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sample_path = repo_root / "examples" / "sample_chunk.json"
    chroma_path = repo_root / "data" / "chroma_db_smoke"

    settings = Settings(
        CHROMA_DB_PATH=str(chroma_path),
        CHROMA_COLLECTION_NAME="hello_world_smoke",
        EMBEDDING_PROVIDER="local",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        RETRIEVAL_K=4,
        SIMILARITY_THRESHOLD=0.2,
    )

    manager = VectorStoreManager(settings=settings)
    chunk = load_sample_chunk(sample_path)

    ingest_result = manager.ingest([chunk])
    print(
        "Ingest result:",
        {
            "ingested": ingest_result.ingested,
            "skipped": ingest_result.skipped,
            "errors": len(ingest_result.errors),
        },
    )

    results = manager.query("what is a neural network")
    print(f"Query returned {len(results)} chunk(s).")

    if results:
        top = results[0]
        print(
            "Top result:",
            {
                "source": top.metadata.source,
                "topic": top.metadata.topic,
                "score": round(top.score, 4),
            },
        )
    else:
        print("No chunk passed the similarity threshold.")


if __name__ == "__main__":
    main()