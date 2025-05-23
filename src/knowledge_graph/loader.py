import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from configs.config import ConfigPath
from knowledge_graph.crud import GraphCrud
from llms.embedding_model import EmbeddingModel
from utils.utils import read_json_file

# --- Constants ---
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_MESH_DEFINITIONS_FILE = "mesh_term_definitions.json"


@dataclass
class PreparedNode:
    temp_id: str
    label: str
    properties: Dict[str, Any]
    embedding: List[float] | None = None


@dataclass
class PreparedRelationship:
    from_temp_id: str
    to_temp_id: str
    rel_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


class GraphLoader:
    """
    Loader for QA_PAIR->CONTEXT->MESH graph schema.

    New schema:
      (QA_PAIR)-[:HAS_CONTEXT]->(CONTEXT{pmid,title,text_content,embedding})
      (CONTEXT)-[:HAS_MESH_TERM]->(MESH)
      (CONTEXT)-[:IS_SIMILAR_TO]->(CONTEXT){score}
    """

    # Node labels
    QA_PAIR_LABEL = "QA_PAIR"
    CONTEXT_LABEL = "CONTEXT"
    MESH_LABEL = "MESH"

    # Relationship types
    HAS_CONTEXT = "HAS_CONTEXT"
    HAS_MESH_TERM = "HAS_MESH_TERM"
    IS_SIMILAR_TO = "IS_SIMILAR_TO"

    # Property keys
    QA_ID = "id"
    QUESTION = "question"
    ANSWER = "answer"
    PMID = "pmid"
    TITLE = "title"
    TEXT = "text_content"
    EMBEDDING_PROP = "embedding"
    MESH_NAME = "name"
    MESH_DEF = "definition"
    SCORE = "score"

    # Vector indexes
    mesh_vector_index_name = "mesh_vector_index"
    context_vector_index_name = "context_vector_index"

    def __init__(
        self,
        data: List[Dict[str, Any]],
        embedding_model: EmbeddingModel,
        crud: GraphCrud,
        mesh_definitions_file: str = DEFAULT_MESH_DEFINITIONS_FILE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        external_data_dir: str = ConfigPath.EXTERNAL_DATA_DIR,
    ):
        self.data = data
        self.embedding_model = embedding_model
        self.crud = crud
        self.similarity_threshold = similarity_threshold

        # Load MESH definitions
        path = os.path.join(external_data_dir, mesh_definitions_file)
        self.mesh_definitions = self._load_mesh_definitions(path)
        self.mesh_node_map: Dict[str, str] = {}

    def _load_mesh_definitions(self, path: str) -> Dict[str, str]:
        try:
            defs = read_json_file(file_path=path)
            print(f"Loaded {len(defs)} MESH definitions from {path}")
            return defs
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Failed to load MESH definitions: {e}")
            return {}

    def load_mesh_nodes(self) -> None:
        """
        Batch create MESH nodes and embed their definitions.
        Populates self.mesh_node_map.
        """
        terms: Set[str] = set()
        for samp in self.data:
            for art in samp.get("articles", []):
                terms.update(art.get("mesh_terms", []))
        if not terms:
            print("No MESH terms found.")
            return

        sorted_terms = sorted(terms)
        definitions = [self.mesh_definitions.get(t, "") for t in sorted_terms]
        embeddings = self.embedding_model.embed_documents(definitions)

        # Create nodes
        props_list = [
            {self.MESH_NAME: t, self.MESH_DEF: d}
            for t, d in zip(sorted_terms, definitions)
        ]
        node_ids = self.crud.create_nodes_batch(
            label=self.MESH_LABEL,
            properties_list=props_list,
        )
        self.mesh_node_map = dict(zip(sorted_terms, node_ids))

        # Set embeddings
        vecs = [
            {"node_id": nid, "embedding": emb} for nid, emb in zip(node_ids, embeddings)
        ]
        self.crud.set_node_vector_properties_batch(
            vectors_data=vecs, property_name=self.EMBEDDING_PROP
        )
        print(f"Loaded {len(node_ids)} MESH nodes.")

        # Ensure mesh vector index
        self.crud.ensure_vector_index(
            index_name=self.mesh_vector_index_name,
            label=self.MESH_LABEL,
            property_name=self.EMBEDDING_PROP,
        )

    def _prepare_qa_contexts(
        self,
    ) -> Tuple[
        List[PreparedNode],
        List[PreparedNode],
        List[PreparedRelationship],
        List[PreparedRelationship],
    ]:
        """
        Build QA_PAIR nodes, CONTEXT nodes (full abstract),
        and QA->CONTEXT, CONTEXT->MESH relationships.
        """
        qa_nodes: List[PreparedNode] = []
        ctx_props: Dict[str, Dict[str, Any]] = {}
        rel_qa_ctx: List[PreparedRelationship] = []
        rel_ctx_mesh: List[PreparedRelationship] = []

        for samp in tqdm(self.data, desc="Preparing QA & Context"):
            qid = samp[self.QA_ID]
            qa_temp = f"qa_{qid}"
            qa_nodes.append(
                PreparedNode(
                    temp_id=qa_temp,
                    label=self.QA_PAIR_LABEL,
                    properties={
                        self.QA_ID: qid,
                        self.QUESTION: samp[self.QUESTION],
                        self.ANSWER: samp[self.ANSWER],
                    },
                )
            )

            for art in samp.get("articles", []):
                pmid = art.get(self.PMID)
                text = art.get("abstract", "")
                title = art.get(self.TITLE, "")
                if not pmid or not text:
                    continue

                ctx_temp = f"ctx_{pmid}"
                if ctx_temp not in ctx_props:
                    ctx_props[ctx_temp] = {
                        self.PMID: pmid,
                        self.TITLE: title,
                        self.TEXT: text,
                    }

                rel_qa_ctx.append(
                    PreparedRelationship(
                        from_temp_id=qa_temp,
                        to_temp_id=ctx_temp,
                        rel_type=self.HAS_CONTEXT,
                    )
                )
                for term in art.get("mesh_terms", []):
                    if term in self.mesh_node_map:
                        rel_ctx_mesh.append(
                            PreparedRelationship(
                                from_temp_id=ctx_temp,
                                to_temp_id=term,
                                rel_type=self.HAS_MESH_TERM,
                            )
                        )
        ctx_nodes = [
            PreparedNode(tid, self.CONTEXT_LABEL, props)
            for tid, props in ctx_props.items()
        ]
        return qa_nodes, ctx_nodes, rel_qa_ctx, rel_ctx_mesh

    def load_qa_contexts(self) -> None:
        """
        Create QA_PAIR and CONTEXT nodes, set embeddings, and relationships.
        """
        if not self.mesh_node_map:
            print("Mesh nodes not loaded. Call load_mesh_nodes() first.")
            return

        # Prepare nodes and relationships
        qa_nodes, ctx_nodes, rel_qa_ctx, rel_ctx_mesh = self._prepare_qa_contexts()

        # Embed contexts
        texts = [n.properties[self.TEXT] for n in ctx_nodes]
        embeddings = self.embedding_model.embed_documents(texts)
        for node, emb in zip(ctx_nodes, embeddings):
            node.embedding = emb

        # Batch create QA_PAIR nodes
        qa_props = [n.properties for n in qa_nodes]
        qa_ids = self.crud.create_nodes_batch(
            label=self.QA_PAIR_LABEL, properties_list=qa_props
        )
        temp_map = {n.temp_id: real for n, real in zip(qa_nodes, qa_ids)}

        # Batch create CONTEXT nodes
        ctx_props_list = [n.properties for n in ctx_nodes]
        ctx_ids = self.crud.create_nodes_batch(
            label=self.CONTEXT_LABEL, properties_list=ctx_props_list
        )
        temp_map.update({n.temp_id: real for n, real in zip(ctx_nodes, ctx_ids)})

        # Set context embeddings
        vec_data = [
            {"node_id": temp_map[n.temp_id], "embedding": n.embedding}
            for n in ctx_nodes
        ]
        self.crud.set_node_vector_properties_batch(
            vectors_data=vec_data, property_name=self.EMBEDDING_PROP
        )
        print("Context embeddings set.")

        # Ensure context vector index
        self.crud.ensure_vector_index(
            index_name=self.context_vector_index_name,
            label=self.CONTEXT_LABEL,
            property_name=self.EMBEDDING_PROP,
        )

        # Create relationships
        rels = []
        for r in rel_qa_ctx + rel_ctx_mesh:
            frm = temp_map.get(r.from_temp_id)
            to = temp_map.get(r.to_temp_id) or self.mesh_node_map.get(r.to_temp_id)
            if frm and to:
                rels.append(
                    {
                        "from_id": frm,
                        "to_id": to,
                        "rel_type": r.rel_type,
                        "properties": r.properties,
                    }
                )
        if rels:
            self.crud.create_relationships_batch(rels)
        print("QA and CONTEXT loading complete.")

    def load_similarities(self) -> None:
        """
        Compute cosine similarities among CONTEXT embeddings,
        and create IS_SIMILAR_TO relationships for scores >= threshold.
        """
        records = self.crud.get_nodes_with_property(
            label=self.CONTEXT_LABEL, property_name=self.EMBEDDING_PROP
        )
        ids = [r["id"] for r in records]
        embs = np.array([r[self.EMBEDDING_PROP] for r in records])
        if embs.shape[0] < 2:
            print("Not enough contexts for similarity.")
            return
        sim_mat = cosine_similarity(embs)
        rows, cols = np.triu_indices(len(ids), k=1)
        rels = []
        for i, j in zip(rows, cols):
            score = float(sim_mat[i, j])
            if score >= self.similarity_threshold:
                rels.append(
                    {
                        "from_id": ids[i],
                        "to_id": ids[j],
                        "rel_type": self.IS_SIMILAR_TO,
                        "properties": {self.SCORE: score},
                    }
                )
        if rels:
            self.crud.create_relationships_batch(rels)
            print(f"Created {len(rels)} similarity relationships.")

    def load_all(self, load_similarities: bool = True) -> None:
        """
        Full pipeline: load MESH, QA_CONTEXT, and optionally similarities.
        """
        start = time.time()
        self.load_mesh_nodes()
        self.load_qa_contexts()
        if load_similarities:
            self.load_similarities()
        print(f"Full load finished in {time.time()-start:.2f}s.")
