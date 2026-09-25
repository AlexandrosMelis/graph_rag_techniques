"""
Entity extraction for the corpus graph.

Two sources: PubMed MeSH headings (curated, biomedical only, one set per passage)
and GLiNER zero-shot NER (works on any text). Both produce (chunk_id, entity, label)
rows; the graph builder deduplicates them.
"""

import re
from collections.abc import Sequence
from typing import Protocol

import pandas as pd

DEFAULT_GLINER_LABELS = (
    "disease",
    "gene",
    "protein",
    "chemical",
    "drug",
    "cell type",
    "organism",
    "anatomical structure",
    "biological process",
)

# MeSH check tags annotate nearly every abstract and would connect everything.
MESH_CHECK_TAGS = frozenset(
    {
        "humans",
        "animals",
        "male",
        "female",
        "adult",
        "middle aged",
        "aged",
        "aged, 80 and over",
        "young adult",
        "adolescent",
        "child",
        "child, preschool",
        "infant",
        "infant, newborn",
        "mice",
        "rats",
        "pregnancy",
    }
)


def clean_mesh_term(term: str) -> str:
    """Drop the qualifier (`/metabolism`) and the major-topic marker (`*`)."""
    return term.split("/")[0].replace("*", "").strip()


def normalize_entity(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text.strip(" .,;:()[]{}\"'")


class EntityExtractor(Protocol):
    name: str

    def extract(self, chunks: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with `chunk_id`, `entity` and `label` columns."""
        ...


class MeshEntityExtractor:
    """Attach each passage's MeSH headings to all of its chunks."""

    name = "mesh"

    def __init__(self, headings: dict[str, Sequence[str]], drop_check_tags: bool = True):
        self.headings = headings
        self.drop_check_tags = drop_check_tags

    def extract(self, chunks: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for chunk_id, pmid in zip(chunks["chunk_id"], chunks["pmid"]):
            for term in self.headings.get(str(pmid), []):
                entity = normalize_entity(clean_mesh_term(term))
                if entity and not (self.drop_check_tags and entity in MESH_CHECK_TAGS):
                    rows.append((chunk_id, entity, "mesh"))
        return pd.DataFrame(rows, columns=["chunk_id", "entity", "label"])


class GlinerEntityExtractor:
    """Zero-shot NER with GLiNER (install the `entities` extra)."""

    name = "gliner"

    def __init__(
        self,
        model_name: str = "urchade/gliner_small-v2.1",
        labels: Sequence[str] = DEFAULT_GLINER_LABELS,
        threshold: float = 0.5,
    ):
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(model_name)
        self.labels = list(labels)
        self.threshold = threshold

    def extract_text(self, text: str) -> list[tuple[str, str]]:
        spans = self.model.predict_entities(text, self.labels, threshold=self.threshold)
        return [(normalize_entity(s["text"]), s["label"]) for s in spans if s["text"].strip()]

    def extract(self, chunks: pd.DataFrame) -> pd.DataFrame:
        from tqdm import tqdm

        rows = []
        for chunk_id, text in tqdm(
            zip(chunks["chunk_id"], chunks["text"]), total=len(chunks), desc="GLiNER"
        ):
            rows.extend((chunk_id, entity, label) for entity, label in self.extract_text(text))
        return pd.DataFrame(rows, columns=["chunk_id", "entity", "label"])
