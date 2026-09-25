"""
Retrieval and end-to-end RAG evaluation. Tunable retrievers are tuned on a dev subset
before the evaluated split is touched; every run is tracked in MLflow.
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

from graph_rag.config import settings
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
from graph_rag.tracking import numeric_metrics, tracked_run

Progress = Optional[Callable[[str], None]]


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


def _load_factory(
    index_dir: Path,
    adapter_dir: str | Path,
    graph_adapter_dir: str | Path,
    reranker_dir: str | Path,
    cross_encoder: Optional[str],
) -> tuple[CorpusIndex, RetrieverFactory]:
    index = CorpusIndex.load(index_dir)
    graph_dir = index_dir / "graph"
    graph = CorpusGraph.load(graph_dir) if graph_dir.exists() else None
    factory = RetrieverFactory(
        index,
        encoder_from_index(index),
        graph,
        adapter_dir=adapter_dir,
        graph_adapter_dir=graph_adapter_dir,
        reranker_dir=reranker_dir,
        cross_encoder=cross_encoder,
    )
    return index, factory


def evaluate_retrieval(
    retrievers: Sequence[str] = ("bm25", "dense", "hybrid"),
    split: str = "dev",
    baseline: str = "dense",
    k_values: Sequence[int] = (1, 5, 10),
    tune_first: bool = False,
    tune_queries: int = 200,
    max_queries: Optional[int] = None,
    index_dir: str | Path | None = None,
    adapter_dir: str | Path = DEFAULT_ADAPTER_DIR,
    graph_adapter_dir: str | Path = DEFAULT_GRAPH_ADAPTER_DIR,
    reranker_dir: str | Path = DEFAULT_RERANKER_DIR,
    cross_encoder: Optional[str] = None,
    output_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
    progress: Progress = None,
) -> dict:
    names = list(dict.fromkeys(retrievers))
    unknown = [n for n in names if n not in RETRIEVERS]
    if unknown:
        raise ValueError(f"Unknown retrievers {unknown}; available: {sorted(RETRIEVERS)}")
    if baseline not in names:
        names.insert(0, baseline)

    splits = load_splits()
    questions = splits[split][:max_queries] if max_queries else splits[split]
    index, factory = _load_factory(
        Path(index_dir or settings.index_dir),
        adapter_dir,
        graph_adapter_dir,
        reranker_dir,
        cross_encoder,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir or settings.results_dir / f"retrieval_{split}_{timestamp}")
    ci_k = max(k_values)

    params = {
        "split": split,
        "retrievers": names,
        "baseline": baseline,
        "k_values": list(k_values),
        "tune": tune_first,
        "queries": len(questions),
    }
    with tracked_run(f"evaluate_retrieval[{split}]", params=params) as parent:
        runs, rows = {}, []
        for name in names:
            tuned = {}
            if tune_first and name in TUNING_GRIDS:
                log(f"tuning {name} on {tune_queries} dev questions")
                tuned, score = tune(factory, name, splits.dev[:tune_queries], k=ci_k, log=log)
                log(f"  best {tuned} (dev recall@{ci_k}={score:.4f})")
            results = run_retrieval(
                questions,
                factory.build(name, **tuned),
                top_k=ci_k,
                output_dir=str(output_dir),
                show_progress=progress is None,
                progress=progress,
            )
            summary = summarize_run(results, k_values, index=index, ci_k=ci_k)
            summary.update(retriever=name, params=tuned, split=split)
            runs[name] = results
            rows.append({"name": name, "summary": summary})

        for row in rows:
            if row["name"] != baseline:
                row["summary"]["vs_baseline"] = {
                    "baseline": baseline,
                    **compare_runs(runs[baseline], runs[row["name"]], k=ci_k),
                }
            save_summary(row["summary"], str(output_dir), row["name"])
            with tracked_run(
                row["name"],
                params={"retriever": row["name"], **row["summary"]["params"]},
                nested=True,
            ) as child:
                child.log_metrics(
                    numeric_metrics(
                        {
                            "metrics": row["summary"]["metrics"],
                            "latency_ms": row["summary"]["latency_ms"],
                            "vs_baseline": row["summary"].get("vs_baseline", {}),
                        }
                    )
                )

        table = format_table(rows, ci_k)
        (output_dir / "summary.md").write_text(table + "\n")
        (output_dir / "config.json").write_text(json.dumps(params, indent=2))
        parent.log_metrics({"recall_ceiling": rows[0]["summary"]["recall_ceiling"]})
        parent.log_artifacts(output_dir, "results")
        log(table)
        log(f"recall ceiling: {rows[0]['summary']['recall_ceiling']:.2%}; results in {output_dir}")
        return {
            "output_dir": str(output_dir),
            "table": table,
            "metrics": {row["name"]: row["summary"]["metrics"] for row in rows},
        }


def evaluate_rag(
    retriever: str = "hybrid",
    split: str = "test",
    n: int = 50,
    top_k: int = 5,
    provider: str = "google",
    model: str = "gemini-2.0-flash-lite",
    seed: int = 42,
    index_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Retrieve, answer with an LLM and score with RAGAS; LLM calls are traced in MLflow."""
    import mlflow

    from graph_rag.evaluation.rag import RAGAnswerer
    from graph_rag.evaluation.ragas_eval import run_evaluation_on_generated_answers
    from graph_rag.llm.chat import ChatModel

    questions = load_splits()[split]
    questions = random.Random(seed).sample(questions, min(n, len(questions)))
    index, factory = _load_factory(
        Path(index_dir or settings.index_dir),
        DEFAULT_ADAPTER_DIR,
        DEFAULT_GRAPH_ADAPTER_DIR,
        DEFAULT_RERANKER_DIR,
        None,
    )
    params = {
        "retriever": retriever,
        "split": split,
        "n": len(questions),
        "top_k": top_k,
        "provider": provider,
        "model": model,
        "seed": seed,
    }
    with tracked_run("evaluate_rag", params=params) as tracker:
        if settings.tracking_enabled:
            mlflow.langchain.autolog()
        else:
            mlflow.tracing.disable()
        llm = ChatModel(provider=provider, model_name=model).initialize_model()
        answerer = RAGAnswerer(factory.build(retriever), llm, index, top_k=top_k)
        generated = []
        for q in questions:
            output = answerer.answer(q.question)
            generated.append(
                {
                    "user_input": q.question,
                    "reference": q.answer,
                    "response": output["response"],
                    "retrieved_contexts": output["retrieved_contexts"],
                }
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = settings.results_dir / f"rag_{retriever}_{split}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        (output_dir / "generated.json").write_text(json.dumps(generated, indent=2))
        result = run_evaluation_on_generated_answers(
            generated, llm=llm, embedding_model=factory.encoder
        )
        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        (output_dir / "ragas_scores.json").write_text(json.dumps(scores, indent=2))
        tracker.log_metrics(scores)
        tracker.log_artifacts(output_dir, "results")
        log(json.dumps(scores, indent=2))
        return {"output_dir": str(output_dir), "scores": scores}
