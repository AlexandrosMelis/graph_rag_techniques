# Roadmap

Where the thesis stands after the September 2026 rewrite, and what is left to do, in order.
Each phase ends with something that can go into the thesis as is.

## Status

| Area | State |
|---|---|
| Data, splits, fixed corpus | done (`graph-rag data prepare`) |
| Index and entity graph (MeSH, GLiNER) | done, not yet built on the full corpus with entities |
| Baselines: BM25, dense, hybrid, cross-encoder | done |
| Graph baselines: PPR, expansion, entity-anchored | done, untuned |
| Query adapter, graph re-ranker, GNN (RQ0) | implemented, never trained on the full corpus |
| Evaluation: metrics, CIs, paired tests, latency, RAG + RAGAS | done |
| Tooling: CLI, MLflow, Temporal, Hub, CI | done |
| Numbers | **none yet**. No result in this repo has been produced with the current code. |

Everything below is ordered by how much it de-risks the thesis.

## Phase 1: first honest numbers (1 to 2 weeks)

Goal: a results table on BioASQ that can go in the thesis, and the RQ0 finding.

- [ ] Build the full index with MeSH entities: `graph-rag index build --entities mesh`. One embedding pass (~30 min CPU, minutes on GPU) plus ~200 NCBI requests for 40k PMIDs. Record entity and mention counts.
- [ ] Baselines on **dev**: `bm25`, `dense`, `hybrid`, `hybrid_ce`. Check the recall ceiling is 100% and note dense vs BM25 vs hybrid.
- [ ] Training-free graph retrievers with tuning: `evaluate retrieval --tune --retrievers ppr,expand,entity`. This answers whether the entity graph adds recall at all before any learning is involved.
- [ ] Query adapter: `train adapter`, then `dense_adapter` on dev. Expect a small gain; the point is that it cannot be worse than the identity (it starts there).
- [ ] Graph re-ranker: `train reranker --edge-types entity,next`, then `graph_reranker` on dev. Compare against `hybrid_ce` (quality ceiling) and `hybrid` (cost floor).
- [ ] RQ0: `train gnn --edge-types knn` vs `--edge-types entity`. Report the cosine-baseline AUC next to each; then `train adapter --space graph` and `graph_space` on dev. This turns the old negative result into a finding: kNN graphs carry no information beyond the embedder.
- [ ] One test-split run at the end with every retriever, `--tune`, and paired tests against `dense`. Do not iterate on test.
- [ ] Repeat training with 3 seeds (`-p seed=`) and report mean ± CI.

Deliverable: results table + RQ0 section. If the graph methods do not beat `hybrid` on BioASQ, that is expected (single-hop questions over abstracts) and the thesis moves to Phase 2 for the queries where graphs should matter.

## Phase 2: multi-hop benchmarks (RQ1, 2 to 3 weeks)

BioASQ mostly needs one passage per answer. The graph hypothesis is about bridging, so the thesis needs datasets where the second hop is measurable.

- [ ] Add a dataset adapter for MuSiQue, 2WikiMultiHopQA and HotpotQA on the corpora used by HippoRAG, so numbers are comparable with published ones. Same `Question` / corpus interface as BioASQ; `data prepare --dataset <name>`.
- [ ] GLiNER entities for these corpora (`--entities gliner`); MeSH does not exist there.
- [ ] Per-query-type breakdown in `evaluate retrieval`: single-hop vs bridge vs comparison, and recall of the *second* gold passage specifically.
- [ ] Edge-type ablation on each dataset: entity only, next only, entity+next, plus kNN. This is the RQ1 table.
- [ ] Add a citation edge type where the corpus provides it (PubMed references via Entrez `elink`).

Deliverable: RQ1 table, which edge types help which query types, with paired tests.

## Phase 3: label-free training and transfer (RQ3, 2 to 3 weeks)

The learned components currently need gold labels from the dataset. For "bring your own corpus" they must train from the corpus alone.

- [ ] Synthetic question generation with a local 7 to 8B model: single-hop questions per chunk, and bridge questions over chunk pairs that share an entity.
- [ ] Consistency filter (keep a question only if its source chunk is retrieved by hybrid search) and cross-encoder margin labels, so other relevant passages are not treated as negatives.
- [ ] Train adapter and re-ranker on synthetic data only; evaluate on the human-labelled dev/test splits.
- [ ] Transfer: train on one corpus, evaluate on another.
- [ ] Router: a small classifier (query length, entity count, dense score margin) deciding dense-only vs graph path, so simple queries pay nothing.

Deliverable: RQ3 result and the router, which is what makes the method usable on an arbitrary corpus.

## Phase 4: efficiency and end-to-end (1 to 2 weeks)

- [ ] Quality vs latency Pareto plot from the p50/p95 numbers already logged: `dense`, `hybrid`, `ppr`, `graph_reranker`, `hybrid_ce`.
- [ ] Indexing cost table: wall time, RAM, index size, LLM tokens (zero for this approach; report what LLM-extraction GraphRAG methods spend for the same corpus).
- [ ] Smaller, faster embedders: EmbeddingGemma-300M and Qwen3-Embedding-0.6B with Matryoshka truncation to 256 dims and int8 vectors; check what retrieval quality costs.
- [ ] HNSW index (faiss or usearch) behind `CorpusIndex.search` once the corpus is larger than ~100k chunks; exhaustive search is fine below that.
- [ ] End-to-end answer quality: `evaluate rag` for `hybrid` vs the best graph retriever on 200 test questions, with one fixed reader model, reporting exact match / F1 alongside RAGAS.

Deliverable: cost chapter.

## Phase 5: writing and reproducibility

- [ ] `docs/` gains one page per research question with the final tables, produced from MLflow runs, not copied by hand.
- [ ] Publish the trained adapter and re-ranker on the Hub (`graph-rag hub push`) and reference them from the README.
- [ ] A `make reproduce` (or Temporal `ExperimentWorkflow`) that regenerates every table from a clean clone.
- [ ] Literature check of 2025 to 2026 GraphRAG papers (HippoRAG 2, LightRAG follow-ups, GNN re-ranking work) before the related-work chapter is final.

## Engineering debt (do when touched)

- `visualization/embeddings.py` and `experiments/exploration.py` are barely covered by tests.
- `graph/crud.py` keeps generic node/relationship helpers that the exporter no longer uses; trim once Neo4j export has been exercised on a real graph.
- The Temporal worker runs activities in a thread pool; GPU training should get its own worker process (`--max-concurrent-activities 1`) so two trainings never share a device.
- `IndexConfig` and the training configs are dataclasses parsed by hand in the CLI; pydantic models would give validation for free.
- Docker images for the worker itself (currently host-only), so a remote GPU box can join the task queue.

## Decisions already made

- The corpus is the dataset's own passage collection, fixed and independent of the evaluated questions. Never rebuild it from gold PMIDs.
- Questions and relevance labels live only in `data/splits/`; nothing about them goes into the index, the graph, or Neo4j.
- Graph edges must carry information the text embedding does not. kNN edges are an ablation, not a feature.
- Retrieval is scored per PMID; chunks collapse to their passage's best rank.
- Dev is for every decision; test is read once per experiment.
- The graph acts at query time on the candidate set (re-ranking / propagation). Projecting queries into a static node-embedding space is the RQ0 ablation only.
