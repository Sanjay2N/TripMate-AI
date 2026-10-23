"""Hybrid retrieval: dense (Qdrant) + BM25, fused with reciprocal rank fusion, then reranked."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from tripmate.config import Settings, get_settings
from tripmate.rag.bm25 import BM25

MODES = ("dense", "bm25", "hybrid")


@dataclass
class Hit:
    doc: dict
    score: float


def load_docs(kb_dir: Path) -> list[dict]:
    docs = []
    for name in ("pois.jsonl", "guides.jsonl", "profiles.jsonl"):
        if not (kb_dir / name).exists():
            continue  # profiles are optional
        with (kb_dir / name).open() as f:
            docs.extend(json.loads(line) for line in f)
    return docs


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: each list votes 1 / (k + rank). Needs no score calibration."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def build_index(settings: Settings | None = None, log=print) -> int:
    """Embed every chunk once and save the vectors to disk.

    Vectors live in a plain .npy file and each process loads them into its own in-memory
    Qdrant. Embedded Qdrant on disk takes an exclusive lock, which would stop the UI, the CLI
    and a long data-generation job from running at the same time.
    """
    s = settings or get_settings()
    docs = load_docs(s.kb_dir)
    embedder = TextEmbedding(s.embed_model)
    vectors = np.array(list(embedder.embed([d["text"] for d in docs], batch_size=64)), "float32")
    s.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(s.index_dir / "vectors.npy", vectors)
    (s.index_dir / "ids.json").write_text(json.dumps([d["id"] for d in docs]))
    log(f"indexed {len(docs)} chunks, dim={vectors.shape[1]} -> {s.index_dir}")
    return len(docs)


def load_index(settings: Settings, docs: dict[str, dict]) -> QdrantClient:
    ids_file = settings.index_dir / "ids.json"
    ids = json.loads(ids_file.read_text()) if ids_file.exists() else []
    if set(ids) != set(docs):
        raise RuntimeError("The vector index is missing or stale. Run: tripmate index")
    vectors = np.load(settings.index_dir / "vectors.npy")
    client = QdrantClient(":memory:")
    client.create_collection(
        settings.collection,
        vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
    )
    payload_keys = ("doc_type", "destination", "kind")
    client.upsert(
        settings.collection,
        points=[
            models.PointStruct(
                id=i, vector=v.tolist(), payload={k: docs[i].get(k) for k in payload_keys}
            )
            for i, v in zip(ids, vectors)
        ],
    )
    return client


class Retriever:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.docs = {d["id"]: d for d in load_docs(self.s.kb_dir)}
        self.embedder = TextEmbedding(self.s.embed_model)
        self.client = load_index(self.s, self.docs)
        self._bm25: dict[tuple[str, str], tuple[list[str], BM25]] = {}
        self._reranker = None

    @property
    def destinations(self) -> list[str]:
        return sorted({d["destination"] for d in self.docs.values()})

    def _dense(self, query, destination, doc_type, kinds, limit) -> list[str]:
        must = [
            models.FieldCondition(key="destination", match=models.MatchValue(value=destination)),
            models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type)),
        ]
        if kinds:
            must.append(models.FieldCondition(key="kind", match=models.MatchAny(any=list(kinds))))
        vector = next(iter(self.embedder.query_embed(query)))
        res = self.client.query_points(
            self.s.collection,
            query=vector.tolist(),
            query_filter=models.Filter(must=must),
            limit=limit,
        )
        return [str(p.id) for p in res.points]

    def _sparse(self, query, destination, doc_type, kinds, limit) -> list[str]:
        key = (destination, doc_type)
        if key not in self._bm25:
            ids = [
                i
                for i, d in self.docs.items()
                if d["destination"] == destination and d["doc_type"] == doc_type
            ]
            self._bm25[key] = (ids, BM25([self.docs[i]["text"] for i in ids]))
        ids, bm25 = self._bm25[key]
        scores = bm25.scores(query)
        ranked = [ids[i] for i in np.argsort(-scores) if scores[i] > 0]
        if kinds:
            ranked = [i for i in ranked if self.docs[i].get("kind") in kinds]
        return ranked[:limit]

    def _rerank(self, query: str, ids: list[str]) -> list[tuple[str, float]]:
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(self.s.rerank_model)
        scores = list(self._reranker.rerank(query, [self.docs[i]["text"] for i in ids]))
        return sorted(zip(ids, map(float, scores)), key=lambda kv: kv[1], reverse=True)

    def search(
        self,
        query: str,
        destination: str,
        doc_type: str = "poi",
        kinds: tuple[str, ...] | None = None,
        k: int = 8,
        mode: str = "hybrid",
        rerank: bool | None = None,
        pool: int = 30,
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        rankings = []
        if mode in ("dense", "hybrid"):
            rankings.append(self._dense(query, destination, doc_type, kinds, pool))
        if mode in ("bm25", "hybrid"):
            rankings.append(self._sparse(query, destination, doc_type, kinds, pool))
        fused = rrf(rankings)
        if self.s.rerank if rerank is None else rerank:
            fused = self._rerank(query, [i for i, _ in fused[:pool]])
        return [Hit(self.docs[i], score) for i, score in fused[:k]]
