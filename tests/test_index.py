import numpy as np
import pandas as pd
import pytest

from graph_rag.index.corpus_index import CorpusIndex, top_k_indices
from graph_rag.index.graph import CorpusGraph, build_corpus_graph
from graph_rag.index.linking import EntityLinker


def test_index_search_and_roundtrip(index, encoder, tmp_path):
    ids, scores = index.search(encoder.encode_query("aspirin inflammation"), top_k=3)
    assert index.pmids[ids[0]] == "1"
    assert np.all(np.diff(scores) <= 0)
    index.save(tmp_path)
    loaded = CorpusIndex.load(tmp_path)
    assert loaded.embedding_model == "fake-bow"
    np.testing.assert_allclose(loaded.embeddings, index.embeddings)


def test_index_has_no_question_data(index, questions):
    texts = set(index.texts.tolist())
    assert not any(q.question in texts or q.answer in texts for q in questions)


def test_gold_coverage_is_the_recall_ceiling(index):
    assert index.gold_coverage([["1", "2"], ["999"]]) == pytest.approx(2 / 3)


def test_top_k_indices_sorted():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert top_k_indices(scores, 2).tolist() == [1, 3]
    assert top_k_indices(scores, 10).tolist() == [1, 3, 2, 0]


def test_duplicate_mentions_become_one_edge(chunks):
    rows = pd.DataFrame(
        {"chunk_id": ["1#0", "1#0", "2#0"], "entity": ["cox", "cox", "cox"], "label": ["mesh"] * 3}
    )
    graph = build_corpus_graph(chunks, rows, max_entity_df=1.0)
    assert graph.mentions.nnz == 2


def test_hub_entities_are_dropped(chunks):
    rows = pd.DataFrame(
        {
            "chunk_id": list(chunks.chunk_id) + ["1#0", "2#0"],
            "entity": ["hub"] * len(chunks) + ["cox", "cox"],
            "label": "mesh",
        }
    )
    graph = build_corpus_graph(chunks, rows, max_entity_df=0.5)
    assert graph.entity_names.tolist() == ["cox"]


def test_chunk_adjacency_is_undirected_and_bounded(graph):
    for edge_types in [("entity",), ("next",), ("knn",), ("entity", "next", "knn")]:
        adj = graph.chunk_adjacency(edge_types)
        assert (adj != adj.T).nnz == 0
        assert adj.diagonal().sum() == 0
        if adj.nnz:
            assert adj.data.min() > 0 and adj.data.max() <= 1.0 + 1e-6


def test_shared_entity_connects_bridge_passages(graph, index):
    adj = graph.chunk_adjacency(("entity",))
    five = int(np.where(index.pmids == "5")[0][0])
    six = int(np.where(index.pmids == "6")[0][0])
    assert adj[five, six] > 0


def test_next_edges_follow_chunk_order():
    chunks = pd.DataFrame(
        {
            "chunk_id": ["a#1", "a#0", "b#0", "a#2"],
            "pmid": ["a", "a", "b", "a"],
            "chunk_index": [1, 0, 0, 2],
            "text": ["x"] * 4,
        }
    )
    graph = build_corpus_graph(chunks, pd.DataFrame(columns=["chunk_id", "entity", "label"]))
    assert sorted(map(tuple, graph.next_edges.T.tolist())) == [(0, 3), (1, 0)]


def test_knn_edges_exclude_self(graph):
    src, dst = graph.knn_edge_index
    assert np.all(src != dst)


def test_bipartite_adjacency_shape(graph):
    adj = graph.bipartite_adjacency()
    n = graph.n_chunks + graph.n_entities
    assert adj.shape == (n, n)
    assert (adj != adj.T).nnz == 0


def test_graph_roundtrip(graph, tmp_path):
    graph.save(tmp_path)
    loaded = CorpusGraph.load(tmp_path)
    assert loaded.entity_names.tolist() == graph.entity_names.tolist()
    assert (loaded.mentions != graph.mentions).nnz == 0
    np.testing.assert_array_equal(loaded.knn_edge_index, graph.knn_edge_index)


def test_linker_matches_multiword_entities(graph, encoder):
    linker = EntityLinker(graph, encoder, embedding_top_k=0)
    links = linker.link("is breast neoplasms risk linked to brca1 protein")
    names = {graph.entity_names[e] for e in links}
    assert {"breast neoplasms", "brca1 protein"} <= names
