"""Shared retrieval implementation used by the API, pipelines, and evals.

Heavy ML dependencies are imported lazily so metric/reporting utilities remain
usable in lightweight environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RetrievalConfig:
    strategy: str = "hybrid"
    rerank: bool = True
    dense_k: int = 8
    bm25_k: int = 8
    rrf_k: int = 60
    top_k: int = 5
    chunk_size: int = 600
    chunk_overlap: int = 100
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __post_init__(self) -> None:
        if self.strategy not in {"dense", "hybrid"}:
            raise ValueError("strategy must be 'dense' or 'hybrid'")
        if self.rerank and self.top_k > self.dense_k + self.bm25_k:
            raise ValueError("top_k cannot exceed the candidate pool")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedChunk:
    rank: int
    chunk_id: str
    document_ids: list[str]
    source: str
    text: str
    score: float | None
    document: Any

    def to_dict(self, include_text: bool = False) -> dict[str, Any]:
        value = {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "document_ids": self.document_ids,
            "source": self.source,
            "score": self.score,
        }
        if include_text:
            value["text"] = self.text
        return value


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_record_index(data_dir: Path) -> list[tuple[str, str]]:
    dataset_path = data_dir / "full_dataset.json"
    if not dataset_path.exists():
        return []
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [
        (str(record["id"]), str(record["question"]).strip())
        for record in records
        if record.get("id") and record.get("question")
    ]


def _attach_stable_metadata(chunks: Iterable[Any], data_dir: Path) -> None:
    record_index = _load_record_index(data_dir)
    for chunk in chunks:
        text = chunk.page_content
        document_ids = [doc_id for doc_id, question in record_index if question in text]
        source = Path(chunk.metadata.get("source", "unknown")).name
        digest = sha256(f"{source}\n{text}".encode("utf-8")).hexdigest()[:16]
        chunk.metadata["chunk_id"] = f"{source}:{digest}"
        chunk.metadata["document_ids"] = document_ids
        chunk.metadata["source_name"] = source.removesuffix(".txt")


def reciprocal_rank_fusion(results: list[list[Any]], rrf_k: int = 60) -> list[Any]:
    scores: dict[str, dict[str, Any]] = {}
    for result_list in results:
        for rank, document in enumerate(result_list, 1):
            chunk_id = document.metadata["chunk_id"]
            entry = scores.setdefault(chunk_id, {"score": 0.0, "document": document})
            entry["score"] += 1.0 / (rrf_k + rank)
    return [
        item["document"]
        for item in sorted(scores.values(), key=lambda value: value["score"], reverse=True)
    ]


class RetrievalEngine:
    def __init__(self, config: RetrievalConfig, data_dir: str | Path | None = None):
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from rank_bm25 import BM25Okapi

        self.config = config
        self.data_dir = Path(data_dir or _project_root() / "data" / "processed").resolve()
        loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["---", "\n\n", "\n", ". "],
        )
        self.chunks = splitter.split_documents(documents)
        _attach_stable_metadata(self.chunks, self.data_dir)

        embedding_model = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": "cpu"},
        )
        vector_size = len(embedding_model.embed_query("dimension probe"))
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name="bloomington",
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name="bloomington",
            embedding=embedding_model,
        )
        self.vector_store.add_documents(self.chunks)

        tokenized = [document.page_content.lower().split() for document in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.reranker = None
        if config.rerank:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(config.reranker_model, max_length=512)

    def _candidates(self, query: str) -> list[Any]:
        vector_results = self.vector_store.similarity_search(query, k=self.config.dense_k)
        if self.config.strategy == "dense":
            return vector_results

        scores = self.bm25.get_scores(query.lower().split())
        indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        bm25_results = [self.chunks[index] for index in indices[: self.config.bm25_k]]
        return reciprocal_rank_fusion(
            [vector_results, bm25_results],
            rrf_k=self.config.rrf_k,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        limit = top_k or self.config.top_k
        candidates = self._candidates(query)
        scored: list[tuple[float | None, Any]]
        if self.reranker is not None:
            pairs = [(query, document.page_content) for document in candidates]
            scores = self.reranker.predict(pairs)
            scored = sorted(
                [(float(score), document) for score, document in zip(scores, candidates)],
                key=lambda value: value[0],
                reverse=True,
            )
        else:
            scored = [(None, document) for document in candidates]

        output = []
        for rank, (score, document) in enumerate(scored[:limit], 1):
            metadata = document.metadata
            output.append(
                RetrievedChunk(
                    rank=rank,
                    chunk_id=metadata["chunk_id"],
                    document_ids=list(metadata.get("document_ids", [])),
                    source=metadata.get("source_name", "unknown"),
                    text=document.page_content,
                    score=score,
                    document=document,
                )
            )
        return output

    def retrieve_legacy(self, query: str, top_k: int | None = None) -> list[tuple[float, Any]]:
        return [
            (item.score if item.score is not None else 0.0, item.document)
            for item in self.retrieve(query, top_k=top_k)
        ]
