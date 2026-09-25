import pandas as pd
import pytest

from graph_rag.data.bioasq import Question, load_corpus, load_questions
from graph_rag.data.chunking import chunk_corpus
from graph_rag.data.entities import MeshEntityExtractor, normalize_entity
from graph_rag.data.pubmed import clean_mesh_term
from graph_rag.data.splits import Splits, assert_disjoint, load_splits, make_splits, save_splits


def _questions(prefix: str, n: int) -> list[Question]:
    return [Question(f"{prefix}{i}", f"{prefix} question {i}", "a", (str(i),)) for i in range(n)]


def test_load_questions_and_corpus(tmp_path):
    pd.DataFrame(
        {
            "question": ["q?"],
            "answer": ["a"],
            "id": [7],
            "relevant_passage_ids": [[11, 12]],
        }
    ).to_parquet(tmp_path / "bioasq_train.parquet")
    pd.DataFrame({"passage": [" text ", "", "other"], "id": [11, 12, 11]}).to_parquet(
        tmp_path / "bioasq_corpus.parquet"
    )
    (q,) = load_questions("train", raw_dir=tmp_path)
    assert q == Question("7", "q?", "a", ("11", "12"))
    corpus = load_corpus(raw_dir=tmp_path)
    assert corpus.to_dict("records") == [{"pmid": "11", "text": "text"}]
    with pytest.raises(ValueError):
        load_questions("dev", raw_dir=tmp_path)


def test_splits_are_deterministic_and_disjoint(tmp_path):
    train, test = _questions("train", 50), _questions("test", 10)
    a = make_splits(train, test, dev_fraction=0.2, seed=1)
    b = make_splits(list(reversed(train)), test, dev_fraction=0.2, seed=1)
    assert [q.id for q in a.dev] == [q.id for q in b.dev]
    assert len(a.dev) == 10 and len(a.train) == 40 and len(a.test) == 10
    save_splits(a, tmp_path)
    assert load_splits(tmp_path).dev == a.dev


def test_overlapping_splits_are_rejected():
    shared = _questions("x", 1)
    with pytest.raises(ValueError, match="both"):
        assert_disjoint(Splits(train=shared, dev=[], test=shared))


def test_chunking_keeps_pmid_and_splits_long_passages():
    corpus = pd.DataFrame({"pmid": ["1", "2"], "text": ["short text", " ".join(["word"] * 300)]})
    chunks = chunk_corpus(corpus, chunk_size=64, chunk_overlap=0)
    assert chunks.loc[0, "chunk_id"] == "1#0"
    long = chunks[chunks.pmid == "2"]
    assert len(long) > 1
    assert long["chunk_index"].tolist() == list(range(len(long)))


def test_mesh_extractor_normalizes_and_drops_check_tags(chunks):
    rows = MeshEntityExtractor(
        {"1": ["Cyclooxygenase*", "Inflammation/metabolism", "Humans"]}
    ).extract(chunks)
    assert set(rows[rows.chunk_id == "1#0"].entity) == {"cyclooxygenase", "inflammation"}


def test_entity_and_mesh_cleaning():
    assert clean_mesh_term("Histone Methyltransferases/*metabolism") == "Histone Methyltransferases"
    assert normalize_entity("  Breast   Neoplasms. ") == "breast neoplasms"
