"""Okapi BM25, written out so every term of the formula is visible.

score(q, d) = sum over query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
"""

import math
import re
from collections import Counter

import numpy as np

STOPWORDS = frozenset(
    "a an and are as at be by for from in is it of on or that the this to with near".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lengths = np.array([sum(tf.values()) for tf in self.tfs], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(docs) else 0.0
        n = len(docs)
        df = Counter(term for tf in self.tfs for term in tf)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tfs))
        if not self.avgdl:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.tfs], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
