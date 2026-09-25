"""
Evaluate retrievers on a question split and compare them against a baseline.

Tunable retrievers (ppr, expand, entity) are tuned on a dev subset first when `--tune`
is given; the test split is only used for the final run.

Usage:
    python -m graph_rag.pipelines.evaluate_retrieval --split dev --retrievers bm25,dense,hybrid
    python -m graph_rag.pipelines.evaluate_retrieval --split test --tune \
        --retrievers dense,hybrid,hybrid_ce,ppr,graph_reranker
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from graph_rag.config import ConfigPath
from graph_rag.data.splits import load_splits
from graph_rag.evaluation.executor import compare_runs, run_retrieval, save_summary, summarize_run
from graph_rag.index.corpus_index import CorpusIndex
from graph_rag.index.graph import CorpusGraph
from graph_rag.retrieval.factory import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_GRAPH_ADAPTER_DIR,
    DEFAULT_RERANKER_DIR,
    RETRIEVERS,
    TUNING_GRIDS,
    RetrieverFactory,
    encoder_from_index,
    tune,
)


def format_table(rows: list[dict], k: int) -> str:
    header = f"| retriever | recall@{k} | ndcg@{k} | mrr@{k} | map@{k} | p50 ms | p95 ms |"
    lines = [header, "|" + "---|" * 7]
    for row in rows:
        m = row["summary"]["metrics"][str(k)]
        lat = row["summary"]["latency_ms"]
        lines.append(
            f"| {row['name']} | {m['recall']:.4f} | {m['ndcg']:.4f} | {m['mrr']:.4f} | "
            f"{m['map']:.4f} | {lat['p50']:.1f} | {lat['p95']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--retrievers", default="bm25,dense,hybrid")
    parser.add_argument("--baseline", default="dense")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune-queries", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--index-dir", default=ConfigPath.INDEX_DIR)
    parser.add_argument("--adapter-dir", default=str(DEFAULT_ADAPTER_DIR))
    parser.add_argument("--graph-adapter-dir", default=str(DEFAULT_GRAPH_ADAPTER_DIR))
    parser.add_argument("--reranker-dir", default=str(DEFAULT_RERANKER_DIR))
    parser.add_argument("--cross-encoder", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    names = [n.strip() for n in args.retrievers.split(",") if n.strip()]
    unknown = [n for n in names if n not in RETRIEVERS]
    if unknown:
        parser.error(f"unknown retrievers {unknown}; available: {sorted(RETRIEVERS)}")
    if args.baseline not in names:
        names.insert(0, args.baseline)

    splits = load_splits()
    questions = splits[args.split][: args.max_queries] if args.max_queries else splits[args.split]
    index = CorpusIndex.load(args.index_dir)
    graph_dir = Path(args.index_dir) / "graph"
    graph = CorpusGraph.load(graph_dir) if graph_dir.exists() else None
    factory = RetrieverFactory(
        index,
        encoder_from_index(index),
        graph,
        adapter_dir=Path(args.adapter_dir),
        graph_adapter_dir=Path(args.graph_adapter_dir),
        reranker_dir=Path(args.reranker_dir),
        cross_encoder=args.cross_encoder,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        ConfigPath.RESULTS_DIR, f"retrieval_{args.split}_{timestamp}"
    )
    ci_k = max(args.k)
    runs, rows = {}, []
    for name in names:
        params = {}
        if args.tune and name in TUNING_GRIDS:
            print(f"tuning {name} on {args.tune_queries} dev questions")
            params, score = tune(factory, name, splits.dev[: args.tune_queries], k=ci_k)
            print(f"  best {params} (dev recall@{ci_k}={score:.4f})")
        retriever = factory.build(name, **params)
        results = run_retrieval(questions, retriever, top_k=ci_k, output_dir=output_dir)
        summary = summarize_run(results, args.k, index=index, ci_k=ci_k)
        summary.update(retriever=name, params=params, split=args.split)
        runs[name] = results
        rows.append({"name": name, "summary": summary})

    for row in rows:
        if row["name"] != args.baseline:
            row["summary"]["vs_baseline"] = {
                "baseline": args.baseline,
                **compare_runs(runs[args.baseline], runs[row["name"]], k=ci_k),
            }
        save_summary(row["summary"], output_dir, row["name"])

    table = format_table(rows, ci_k)
    Path(output_dir, "summary.md").write_text(table + "\n")
    Path(output_dir, "config.json").write_text(json.dumps(vars(args), indent=2))
    print(table)
    print(
        f"recall ceiling (gold passages present in the index): {rows[0]['summary']['recall_ceiling']:.2%}"
    )
    print(f"results written to {output_dir}")


if __name__ == "__main__":
    main()
