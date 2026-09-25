# Graph RAG Techniques

**AUTH Diploma Thesis Project**

Graph-augmented retrieval for biomedical RAG. The question this code is built to answer: *when does a corpus graph improve retrieval over strong dense and lexical search, and what does it cost?*

A graph only helps when its edges carry information the text embedding does not (shared entities, document structure) and when the question needs it (bridging between passages). Everything here is set up to measure that without fooling ourselves: a fixed corpus, question-level splits, no relevance labels anywhere in the index, strong baselines and paired significance tests.

## Research questions

| | Question |
|---|---|
| RQ0 | Do embedding-similarity (kNN) graphs add anything beyond the embedder? Their edges are a function of the node features, so link prediction on them is solved by cosine similarity alone. |
| RQ1 | Which edge types (entity co-mention, same-passage adjacency, kNN) help which queries? |
| RQ2 | Does a query-conditioned graph re-ranker beat training-free propagation (PPR), and how close does it get to a cross-encoder at what latency? |
| RQ3 | Can graph components be trained without human labels and transfer across corpora? (planned) |

## How it works

### Pipeline

```mermaid
flowchart LR
    subgraph data["prepare_data"]
        HF[("rag-mini-bioasq<br/>pinned revision")] --> Q["questions<br/>train / test"]
        HF --> C["corpus<br/>40,181 passages"]
        Q --> S["splits<br/>train / dev / test"]
    end
    subgraph index["build_index"]
        C --> CH["chunks"] --> EMB["embeddings"]
        CH --> ENT["entities<br/>MeSH or GLiNER"]
        EMB --> IDX[("CorpusIndex")]
        ENT --> G[("CorpusGraph")]
    end
    subgraph train["train_*"]
        S --> AD["query adapter"]
        S --> RR["graph re-ranker"]
        G --> GNN["GNN node embeddings"]
    end
    subgraph eval["evaluate_*"]
        IDX --> R["retrievers"]
        G --> R
        AD --> R
        RR --> R
        GNN --> R
        R --> M["IR metrics, CIs,<br/>paired tests, latency"]
        R --> RAG["LLM answer +<br/>RAGAS"]
    end
```

The index and graph live in memory as NumPy / SciPy arrays, so a query never waits on a database. Neo4j is an optional export for exploring and visualizing the graph.

### Query time

```mermaid
flowchart TD
    q["question"] --> enc["encoder<br/>(query prompt)"]
    q --> bm25["BM25"]
    q --> link["entity linker"]
    enc --> dense["dense search"]
    dense --> rrf["RRF fusion<br/>top-K candidates"]
    bm25 --> rrf
    rrf --> ppr["personalized PageRank<br/>over chunk-entity graph"]
    link --> ppr
    rrf --> gr["graph re-ranker<br/>GATv2 on candidate subgraph"]
    link --> gr
    rrf --> ce["cross-encoder<br/>(quality upper bound)"]
    ppr --> out["ranked chunks -> PMIDs"]
    gr --> out
    ce --> out
```

The graph re-ranker builds a small graph over the K candidates, with edges from shared entities and same-passage adjacency. Its node inputs depend on the query (q ⊙ d, dense score, first-stage rank, linked-entity coverage), so message passing spreads relevance between related candidates. Candidates are built the same way at training and inference time, and gold labels are only training targets.

### Corpus graph

```mermaid
flowchart LR
    c1["CHUNK<br/>chunk_id, pmid, text, embedding"] -- "MENTIONS {weight = IDF}" --> e1["ENTITY<br/>name, label"]
    c2["CHUNK"] -- MENTIONS --> e1
    c1 -- NEXT --> c3["CHUNK<br/>(same passage)"]
    c1 -. "SIMILAR_TO {score}<br/>ablation only" .-> c2
```

There are no question nodes and no relevance edges: questions and their gold passages exist only in the split files. Each (chunk, entity) pair is stored once. Entities that appear in more than 5% of chunks are dropped because they connect everything, and so are MeSH check tags such as *Humans*.

### Evaluation protocol

```mermaid
flowchart LR
    otr["official train<br/>4,012 questions"] --> tr["train 90%<br/>fit models"]
    otr --> dv["dev 10%<br/>checkpoint selection,<br/>hyperparameter tuning"]
    ote["official test<br/>707 questions"] --> te["test<br/>final numbers only"]
```

- The corpus is the dataset's own passage collection and doesn't depend on which questions are evaluated. Every run reports the recall ceiling, which is the share of gold passages present in the index.
- Retrieval is scored per PMID. Chunk hits collapse to the passage's best rank.
- Metrics are Recall, Precision, nDCG, MRR, MAP, Success and Coverage at k, with bootstrap confidence intervals, paired bootstrap tests against a baseline, and p50/p95 latency.
- A retriever error stops the run instead of being scored as an empty result.

## Retrievers

| Name | What it does | Training |
|---|---|---|
| `bm25` | BM25 over chunk text (bm25s) | none |
| `dense` | cosine search over chunk embeddings | none |
| `hybrid` | BM25 + dense with reciprocal rank fusion | none |
| `hybrid_ce` | hybrid, re-ranked by a cross-encoder (default `ncbi/MedCPT-Cross-Encoder`) | none |
| `dense_adapter` | dense with a low-rank residual query adapter (identity at init, ~98k params) | train split |
| `ppr` | personalized PageRank over the chunk-entity graph, seeded by hybrid hits and query entities | tuned on dev |
| `expand` | hybrid, then relevance propagation to graph neighbours | tuned on dev |
| `entity` | dense search restricted to chunks mentioning query entities | tuned on dev |
| `graph_reranker` | query-conditioned GATv2 over the hybrid candidates | train split |
| `graph_space` | search in GNN node-embedding space through a graph-space adapter (RQ0) | train split |

## Repository layout

```
graph_rag_techniques/
├── pyproject.toml / uv.lock      # dependencies (uv), tool config
├── src/graph_rag/
│   ├── config.py                 # env vars (checked lazily) and data paths
│   ├── data/                     # dataset download, splits, chunking, PubMed client, entity extraction
│   ├── index/                    # CorpusIndex, CorpusGraph, entity linker
│   ├── retrieval/                # retrievers + factory (named retrievers, dev tuning)
│   ├── models/                   # query adapter, graph re-ranker, losses
│   ├── gnn/                      # GNN node encoder (link prediction) for the RQ0 experiments
│   ├── evaluation/               # metrics, runner, paired tests, RAG answerer, RAGAS
│   ├── graph/                    # Neo4j connection and exporter (exploration only)
│   ├── llm/                      # embedding model, cross-encoder, chat models
│   ├── visualization/            # t-SNE of text vs graph embeddings
│   ├── experiments/              # exploration helpers
│   └── pipelines/                # entry points, run with `python -m`
└── tests/                        # offline tests on a toy corpus with a fake encoder
```

## Setup

Requirements: Python 3.10 to 3.12 and [uv](https://docs.astral.sh/uv/). A GPU (CUDA or Apple MPS) speeds up embedding but isn't required. Neo4j is only needed for `--export-neo4j`.

```bash
uv sync                          # add `--extra entities` for GLiNER
cp .env.example .env             # ENTREZ_EMAIL is needed for MeSH entities
uv run pytest
```

## Running

```bash
# 1. data: pinned download + train/dev/test splits
uv run python -m graph_rag.pipelines.prepare_data

# 2. index: chunks, embeddings, entity graph (MeSH via NCBI Entrez, cached and resumable)
uv run python -m graph_rag.pipelines.build_index --entities mesh
#    other embedders: --embedding-model Qwen/Qwen3-Embedding-0.6B --query-prompt-name query
#    RQ0 ablation edges: --knn-k 10      general-domain entities: --entities gliner

# 3. learned components (train split, checkpoint picked on dev)
uv run python -m graph_rag.pipelines.train_query_adapter
uv run python -m graph_rag.pipelines.train_graph_reranker --edge-types entity,next
uv run python -m graph_rag.pipelines.train_gnn --edge-types entity        # RQ0: compare with --edge-types knn

# 4. evaluation (use --split dev while iterating; test once at the end)
uv run python -m graph_rag.pipelines.evaluate_retrieval --split dev \
    --retrievers bm25,dense,hybrid,dense_adapter,ppr,graph_reranker
uv run python -m graph_rag.pipelines.evaluate_retrieval --split test --tune \
    --retrievers dense,hybrid,hybrid_ce,dense_adapter,ppr,expand,entity,graph_reranker
uv run python -m graph_rag.pipelines.evaluate_rag --retriever hybrid --n 50
```

Results go to `data/results/<run>/`: `summary.md` (comparison table), one `*_summary.json` per retriever (metrics, confidence intervals, latency, recall ceiling, paired test against the baseline) and per-query CSVs.

`train_gnn` prints the zero-parameter baseline next to the GNN, i.e. the AUC of cosine(x_i, x_j) on the same held-out edges. On kNN edges that baseline is about 1.0, and the GNN can't learn anything the embedding didn't already contain.

## Environment

| Variable | Needed for |
|---|---|
| `ENTREZ_EMAIL`, `ENTREZ_API_KEY` (optional, raises the rate limit) | `build_index --entities mesh` |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_PUBMED_DATABASE` | `build_index --export-neo4j` |
| `GOOGLE_API_KEY` / `GROQ_API_KEY` | `evaluate_rag` |

## Data

Questions and passages come from [`enelpol/rag-mini-bioasq`](https://huggingface.co/datasets/enelpol/rag-mini-bioasq) (CC BY 2.5), pinned to a fixed revision. MeSH headings come from PubMed through NCBI Entrez. Everything under `data/` is generated locally and git-ignored.
