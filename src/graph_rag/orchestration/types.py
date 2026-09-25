"""
Workflow and activity inputs. Plain dataclasses with JSON-friendly fields, kept free of
heavy imports because the workflow sandbox re-imports them. Component settings travel as
override dicts so adding a training option does not change the workflow input shape.
"""

from dataclasses import dataclass, field

TRAINABLE = ("gnn", "adapter", "graph_adapter", "reranker")


@dataclass
class PrepareDataInput:
    dev_fraction: float = 0.1
    seed: int = 42
    force_download: bool = False


@dataclass
class IndexInput:
    overrides: dict = field(default_factory=dict)  # IndexConfig fields


@dataclass
class IndexCorpusInput:
    prepare: PrepareDataInput = field(default_factory=PrepareDataInput)
    index: IndexInput = field(default_factory=IndexInput)


@dataclass
class TrainInput:
    component: str  # one of TRAINABLE
    overrides: dict = field(default_factory=dict)  # training config fields
    edge_types: str = ""


@dataclass
class EvaluateInput:
    split: str = "dev"
    retrievers: list[str] = field(default_factory=lambda: ["bm25", "dense", "hybrid"])
    baseline: str = "dense"
    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])
    tune: bool = False


@dataclass
class ExperimentInput:
    train: list[str] = field(default_factory=lambda: ["adapter", "reranker"])
    gnn: TrainInput = field(default_factory=lambda: TrainInput("gnn", edge_types="entity"))
    adapter: TrainInput = field(default_factory=lambda: TrainInput("adapter"))
    graph_adapter: TrainInput = field(default_factory=lambda: TrainInput("graph_adapter"))
    reranker: TrainInput = field(
        default_factory=lambda: TrainInput("reranker", edge_types="entity,next")
    )
    evaluate: EvaluateInput = field(default_factory=EvaluateInput)
