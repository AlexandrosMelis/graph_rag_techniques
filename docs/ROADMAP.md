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

Everything below is ordered by how much it de-risks the thesis. Phase 0 is the generalisation of the framework itself and runs first, so that every experiment after it is produced by corpus-agnostic code.

## Phase 0: corpus profiles, so biomedical is one case, not the framework (2 to 3 weeks)

The goal of the project is a general graph-augmented retrieval framework that a community can use on its own corpus. BioASQ / PubMed is the first case study, but today it leaks into the code in at least these places:

| Hardcoded now | Where |
|---|---|
| One dataset, one HF repo, one file layout | `data/bioasq.py`, `pipelines/prepare_data.py` |
| `pmid` as the document id (92 occurrences across 18 modules) | `Hit`, `CorpusIndex`, `Question`, evaluation result keys |
| MeSH entities via NCBI, biomedical GLiNER labels, MeSH check tags | `data/entities.py`, `data/pubmed.py`, `pipelines/build_index.py` |
| PubMedBERT and MedCPT defaults | `llm/embeddings.py`, `llm/reranker.py` |
| "You are a biomedical expert" prompt | `evaluation/rag.py` |
| `--entities mesh\|gliner\|none` as a closed list | CLI, `IndexConfig` |
| One `data/` tree, so one corpus per checkout | `config.py` |

This phase comes **before** Phase 1: the experiments should be run once, on the general code, so BioASQ numbers are produced the same way as everything that follows.

### Design: a corpus profile is a YAML file

Everything corpus-specific lives in one declarative profile. Built-in profiles ship in `src/graph_rag/profiles/` (`bioasq.yaml`, `beir-scifact.yaml`, `beir-fiqa.yaml`, `local-jsonl.yaml` as a template); users point at their own with `--corpus path/to/profile.yaml`. The Python code never mentions PubMed, MeSH or PMIDs outside the plugins that implement them.

```yaml
name: bioasq
description: BioASQ questions over PubMed abstracts (rag-mini-bioasq)

source:                      # where documents and questions come from
  type: huggingface          # huggingface | beir | local
  repo_id: enelpol/rag-mini-bioasq
  revision: 8a845907dc1cff31d42fa6f7bb9c6eef5f3ae6f6   # pinned, always
  corpus:
    file: text-corpus/test-00000-of-00001.parquet
    id_field: id
    text_field: passage
    title_field: null        # prepended to the first chunk when set
    metadata_fields: []      # kept alongside the corpus (tags, authors, year, citations...)
  questions:                 # optional; a corpus can be indexed and queried without any
    splits:
      train: question-answer-passages/train-00000-of-00001.parquet
      test: question-answer-passages/test-00000-of-00001.parquet
    id_field: id
    question_field: question
    answer_field: answer     # optional (BEIR has none: retrieval-only evaluation)
    relevant_ids_field: relevant_passage_ids

splits:
  dev_fraction: 0.1          # carved from train when the source has no dev split
  seed: 42

chunking:
  size: 384
  overlap: 64
  prepend_title: true

embedding:
  model: neuml/pubmedbert-base-embeddings
  query_prompt_name: null    # "query" for Qwen3-Embedding / EmbeddingGemma
  query_prefix: ""           # "query: " for E5-style models
  document_prefix: ""

reranker:
  cross_encoder: ncbi/MedCPT-Cross-Encoder

entities:                    # ordered list of extractor plugins; rows are unioned
  - type: mesh               # built-in: PubMed MeSH headings via NCBI (needs ENTREZ_EMAIL)
    drop_check_tags: true
  - type: gliner             # built-in: zero-shot NER, any domain
    labels: [disease, gene, protein, chemical, drug]
    threshold: 0.5
  # - type: metadata         # built-in: entities from a metadata column (tags, categories, authors)
  #   field: tags
  # - type: my_package.extractors:PatentClassifier   # any importable class with .extract(chunks)
  stop_entities: []          # dropped after normalisation
  max_entity_df: 0.05        # hub entities above this document frequency are dropped

graph:
  edges: [entity, next]      # entity | next | link | knn
  link_fields: []            # metadata fields holding ids of related documents (citations, "see also")
  knn_k: 0                   # embedding kNN edges, ablation only

generation:
  system_prompt: "You are a biomedical expert. Answer using only the numbered context passages."
```

A BEIR profile is the same schema with `source.type: beir` and `repo_id: BeIR/scifact`: the loader knows the BEIR layout (`corpus`/`queries` configs, `<repo>-qrels` with `train.tsv`/`dev.tsv`/`test.tsv`, columns `_id`, `title`, `text`, `query-id`, `corpus-id`, `score`). That single loader covers scifact, nfcorpus, fiqa, hotpotqa, nq, msmarco and the rest of the benchmark. `source.type: local` reads parquet / jsonl / csv files from disk with the same field mapping, which is the "bring your own corpus" path.

### Work items

- [ ] `data/profile.py`: `CorpusProfile` pydantic model, YAML loading, built-in profile lookup by name, validation errors that name the field.
- [ ] `data/sources/`: `huggingface` (pinned files with a field mapping), `beir`, `local`. Each returns the same two things: a corpus DataFrame (`doc_id`, `text`, `title`, metadata columns) and `Question` lists per available split. Splits logic handles any subset of train/dev/test (dev carved from train when missing; no train means evaluation only, with a clear warning).
- [ ] Rename `pmid` to `doc_id` everywhere: `Hit`, `CorpusIndex.doc_ids`, `Question.relevant_ids`, result keys `true_ids` / `retrieved_ids`, split files. Nothing has been generated with the current names, so no migration is needed.
- [ ] Entity extractor registry: `type` resolves to a built-in (`mesh`, `gliner`, `metadata`) or a dotted import path; each extractor is built from its profile block plus the corpus DataFrame and the corpus paths (the MeSH one needs the cache directory). Move MeSH and PubMed code under `data/plugins/pubmed/` so the biomedical case is visibly a plugin.
- [ ] `link` edge type in `CorpusGraph`: doc-to-doc references from `graph.link_fields`, mapped to chunk-chunk edges, exported to Neo4j as `LINKS_TO`.
- [ ] Profile-driven defaults: embedding model, query/document prompts, cross-encoder, chunking, graph edges and RAG prompt all read from the profile; CLI flags only override. Generic profile defaults use general-domain models (`BAAI/bge-small-en-v1.5` / `BAAI/bge-reranker-v2-m3`), not the biomedical ones.
- [ ] Per-corpus data layout: `data/<corpus>/{raw,splits,index,models,results}`; `GRAPH_RAG_CORPUS` env var and a global `graph-rag --corpus NAME_OR_PATH` option select the profile; MLflow runs tagged with the corpus; Temporal inputs carry the profile reference.
- [ ] `graph-rag corpus validate <profile>`: loads the profile, downloads or opens the source, prints document/question counts, id overlap between splits, gold coverage, and which entity extractors are runnable with the current environment.
- [ ] Tests: profile parsing and validation, each source type on tiny fixture files, the `metadata` extractor, `link` edges, and one end-to-end `data prepare` + `index build` + `evaluate retrieval` on a 20-document local corpus with no biomedical dependency installed.
- [ ] Docs: README repositioned as a general framework with BioASQ and SciFact as the two worked examples; a "Bring your own corpus" page with the local-files profile walkthrough; a "Writing an entity extractor" page.

Deliverable: `graph-rag --corpus beir-scifact data prepare && graph-rag --corpus beir-scifact index build && graph-rag --corpus beir-scifact evaluate retrieval` runs without touching any biomedical code, and the BioASQ profile produces the same index as today.

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

- The corpus is the dataset's own document collection, fixed and independent of the evaluated questions. Never rebuild it from gold document ids.
- Corpus-specific knowledge (ids, entity sources, models, prompts) lives in a profile file or a plugin, never in the core package. Biomedical is one profile among several.
- Questions and relevance labels live only in `data/splits/`; nothing about them goes into the index, the graph, or Neo4j.
- Graph edges must carry information the text embedding does not. kNN edges are an ablation, not a feature.
- Retrieval is scored per PMID; chunks collapse to their passage's best rank.
- Dev is for every decision; test is read once per experiment.
- The graph acts at query time on the candidate set (re-ranking / propagation). Projecting queries into a static node-embedding space is the RQ0 ablation only.
