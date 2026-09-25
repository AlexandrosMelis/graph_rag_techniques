import torch
from neo4j import GraphDatabase
from torch_geometric.data import Data


class QueryGATLoader:
    """
    For a given query embedding, retrieve:
      - top_k contexts by semantic sim
      - their neighbors via IS_SIMILAR_TO
    and build a PyG Data object with:
      - node 0: the query
      - nodes 1…N: the contexts
      - edges: (0->i) for each context, plus their inter‐context edges
    """

    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password), database=database
        )

    def build_subgraph(self, q_emb: torch.Tensor, top_k: int = 10) -> Data:
        """
        :param q_emb: torch.Tensor[d_sem]  (CPU or GPU – we .tolist() it)
        """
        neo4j_q = q_emb.tolist()
        cypher = """
        WITH $q_emb AS q_emb
        // get top-k contexts by semantic similarity
        MATCH (c:CONTEXT)
        WITH c, gds.similarity.cosine(q_emb, c.embedding) AS sim
        ORDER BY sim DESC
        LIMIT $k
        // fetch their neighbors
        OPTIONAL MATCH (c)-[:IS_SIMILAR_TO]->(nbr:CONTEXT)
        RETURN
          collect(DISTINCT id(c))             AS ctxt_ids,
          collect(DISTINCT c.embedding)      AS ctxt_embs,
          collect(DISTINCT id(nbr))          AS nbr_ids,
          collect(DISTINCT nbr.embedding)    AS nbr_embs,
          collect(DISTINCT [id(c), id(nbr)]) AS edges
        """
        with self.driver.session() as sess:
            rec = sess.run(cypher, q_emb=neo4j_q, k=top_k).single()

        ctxt_ids = rec["ctxt_ids"]
        ctxt_embs = rec["ctxt_embs"]
        nbr_ids = [nid for nid in rec["nbr_ids"] if nid is not None]
        nbr_embs = [emb for emb in rec["nbr_embs"] if emb is not None]
        raw_edges = [pair for pair in rec["edges"] if pair[1] is not None]

        # Build node list: query node id = -1, then contexts, then neighbors
        node_ids = [-1] + ctxt_ids + nbr_ids
        emb_list = [q_emb.tolist()] + ctxt_embs + nbr_embs

        # map to indices
        id2idx = {nid: i for i, nid in enumerate(node_ids)}

        # build edge_index: query->contexts
        edges = []
        for cid in ctxt_ids:
            edges.append([id2idx[-1], id2idx[cid]])
            edges.append([id2idx[cid], id2idx[-1]])  # undirected

        # inter‐context/neighbors edges
        for src, dst in raw_edges:
            if dst in id2idx:
                edges.append([id2idx[src], id2idx[dst]])
                edges.append([id2idx[dst], id2idx[src]])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(emb_list, dtype=torch.float)

        return Data(x=x, edge_index=edge_index)
