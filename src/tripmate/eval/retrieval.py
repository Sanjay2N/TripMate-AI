"""Retrieval ablation: hit@k and MRR for each retrieval mode over a hand-labeled gold set."""

import json
from pathlib import Path

from tripmate.rag.retriever import Retriever

CONFIGS = [
    ("dense", "dense", False),
    ("bm25", "bm25", False),
    ("hybrid (rrf)", "hybrid", False),
    ("hybrid + rerank", "hybrid", True),
]


def evaluate(gold_file: Path, k: int = 5, retriever: Retriever | None = None) -> list[dict]:
    r = retriever or Retriever()
    gold = [json.loads(line) for line in gold_file.read_text().splitlines() if line.strip()]
    rows = []
    for label, mode, rerank in CONFIGS:
        hits = rr = 0.0
        for g in gold:
            names = [
                h.doc["name"]
                for h in r.search(g["query"], g["destination"], k=k, mode=mode, rerank=rerank)
            ]
            rank = next((i for i, n in enumerate(names, 1) if n in g["relevant"]), None)
            if rank:
                hits += 1
                rr += 1 / rank
        rows.append({"config": label, f"hit@{k}": hits / len(gold), "mrr": rr / len(gold)})
    return rows


def format_table(rows: list[dict]) -> str:
    keys = list(rows[0])
    lines = ["  ".join(f"{k:<18}" for k in keys)]
    for row in rows:
        lines.append(
            "  ".join(
                f"{row[k]:<18.3f}" if isinstance(row[k], float) else f"{row[k]:<18}" for k in keys
            )
        )
    return "\n".join(lines)
