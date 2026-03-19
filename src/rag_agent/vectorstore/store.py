"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        # TODO: implement
        # 1. Ensure Path(self._settings.chroma_db_path).mkdir(parents=True, exist_ok=True)
        from pathlib import Path
        import chromadb
        Path(self._settings.chroma_db_path).mkdir(parents=True, exist_ok=True)
        # 2. chromadb.PersistentClient(path=self._settings.chroma_db_path)
        self._client = chromadb.PersistentClient(
            path=self._settings.chroma_db_path
        )
        # 3. client.get_or_create_collection(
        #        name=self._settings.chroma_collection_name,
        #        metadata={"hnsw:space": "cosine"}   # cosine similarity
        #    )
        self._collection = self._client.get_or_create_collection(
            name=self._settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        # 4. Log successful initialisation with collection name and item count
        count = self._collection.count()
        logger.info(
            f"ChromaDB initialised: {self._settings.chroma_collection_name} | "
            f"Items: {count}"
        )

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        # TODO: implement
        # self._collection.get(ids=[chunk_id])
        result = self._collection.get(ids=[chunk_id])
        # Return True if the result contains the ID, False otherwise
        if result and result.get("ids"):
            return len(result["ids"]) > 0
        return False
        

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        # TODO: implement
        # result = IngestionResult()
        # For each chunk:
        #   - check_duplicate(chunk.chunk_id) → if True, result.skipped += 1, continue
        #   - embed chunk.chunk_text using self._embeddings.embed_documents([chunk.chunk_text])
        #   - self._collection.upsert(
        #         ids=[chunk.chunk_id],
        #         embeddings=[embedding],
        #         documents=[chunk.chunk_text],
        #         metadatas=[chunk.metadata.to_dict()]
        #     )
        #   - result.ingested += 1
        # Log summary and return result
        result = IngestionResult()

        batch_size = 100

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            ids = []
            texts = []
            metadatas = []

            for chunk in batch:
                try:
                    if self.check_duplicate(chunk.chunk_id):
                        result.skipped += 1
                        continue

                    ids.append(chunk.chunk_id)
                    texts.append(chunk.text)
                    metadatas.append(chunk.metadata.to_dict())

                except Exception as e:
                    logger.error(f"Error preparing chunk: {e}")
                    result.errored += 1

            if not texts:
                continue

            try:
                embeddings = self._embeddings.embed_documents(texts)

            except Exception as e:
                logger.error(f"Error embedding documents: {e}")
                result.errored += len(texts)
                continue

                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )

                result.ingested += len(ids)

            except Exception as e:
                logger.error(f"Batch ingestion error: {e}")
                result.errored += len(ids)

        logger.info(
            f"Ingestion complete | Ingested: {result.ingested}, "
            f"Skipped: {result.skipped}, Errors: {result.errored}"
        )

        return result
        

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        # TODO: implement
        # k = k or self._settings.retrieval_k
        # Build where_filter dict from topic_filter and difficulty_filter if provided
        # Embed query_text using self._embeddings.embed_query(query_text)
        # self._collection.query(
        #     query_embeddings=[query_embedding],
        #     n_results=k,
        #     where=where_filter,      # None if no filters
        #     include=["documents", "metadatas", "distances"]
        # )
        # Convert distances to similarity scores: score = 1 - distance (for cosine)
        # Filter out chunks below self._settings.similarity_threshold
        # Return list of RetrievedChunk objects sorted by score descending
        k = k or self._settings.retrieval_k

        where_filter = {}

        if topic_filter:
            where_filter["topic"] = topic_filter

        if difficulty_filter:
            where_filter["difficulty"] = difficulty_filter

        if not where_filter:
            where_filter = None

        query_embedding = self._embeddings.embed_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []

        for i in range(len(results["documents"][0])):
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            score = 1 - distance  # cosine similarity

            if score < self._settings.similarity_threshold:
                continue

            retrieved.append(
                RetrievedChunk(
                    text=text,
                    metadata=ChunkMetadata(**metadata),
                    score=score
                )
            )

        # sort by score descending
        retrieved.sort(key=lambda x: x.score, reverse=True)

        return retrieved
    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        # TODO: implement
        # Query all metadata from the collection
        # Group by metadata["source"] and count chunks per source
        # Return sorted list of dicts
        results = self._collection.get(include=["metadatas"])
        docs = {}

        for meta in results["metadatas"]:
            source = meta["source"]

            if source not in docs:
                docs[source] = {
                    "source": source,
                    "topic": meta.get("topic", "unknown"),
                    "chunk_count": 0
                }
            docs[source]["chunk_count"] += 1
        return list(docs.values())

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        # TODO: implement
        # self._collection.get(where={"source": source}, include=["documents", "metadatas"])
        # Reconstruct DocumentChunk objects from results
        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"]
        )

        chunks = []

        for text, meta in zip(results["documents"], results["metadatas"]):
            chunks.append(
                DocumentChunk(
                    chunk_id="",
                    text=text,
                    metadata=ChunkMetadata(**meta)
                )
            )
        return chunks

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        # TODO: implement
        results = self._collection.get(include=["metadatas"])
        total_chunks = len(results["metadatas"])
        topics = set()
        sources = set()
        bonus_present = False

        for meta in results["metadatas"]:
            topics.add(meta.get("topic"))
            sources.add(meta.get("source"))

            if meta.get("is_bonus"):
                bonus_present = True

        return {
            "total_chunks": total_chunks,
            "topics": list(topics),
            "sources": list(sources),
            "bonus_topics_present": bonus_present
        }
        

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        # TODO: implement
        # self._collection.delete(where={"source": source})
        results = self._collection.get(where={"source": source})
        count = len(results["ids"]) if results and results.get("ids") else 0
        self._collection.delete(where={"source": source})
        return count
    