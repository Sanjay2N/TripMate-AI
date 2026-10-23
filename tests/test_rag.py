from tripmate.rag.bm25 import BM25, tokenize
from tripmate.rag.retriever import rrf


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("The Fort, near the Lake!") == ["fort", "lake"]


def test_bm25_ranks_matching_doc_first_and_rare_terms_higher():
    docs = ["beach with water sports", "old fort on a hill", "beach shack serving seafood"]
    bm25 = BM25(docs)
    assert bm25.scores("fort").argmax() == 1
    # "seafood" appears in one doc, "beach" in two: the rarer term must weigh more
    assert bm25.idf["seafood"] > bm25.idf["beach"]
    assert bm25.scores("zzz").sum() == 0


def test_rrf_rewards_agreement_between_lists():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"], ["b", "x"]])
    assert fused[0][0] == "b"
    assert {doc for doc, _ in fused} == {"a", "b", "c", "d", "x"}
