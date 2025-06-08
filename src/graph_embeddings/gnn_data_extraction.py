import pandas as pd
import torch
from graphdatascience import GraphDataScience

# ----------------------------------
# Data Extraction
# ----------------------------------


def connect_to_neo4j(
    uri: str, user: str, password: str, database: str
) -> GraphDataScience:
    """Create a GDS client connected to the specified Neo4j database."""
    return GraphDataScience(uri, auth=(user, password), database=database)


def sample_graph(
    gds: GraphDataScience, graph_name: str, sample_name: str, seed: int = 42
) -> GraphDataScience:
    """Sample the input graph via random walk with restarts."""

    if gds.graph.exists(sample_name)["exists"]:
        print(f"Graph '{sample_name}' already exists. Fetch it.")
        sampled = gds.graph.get(sample_name)
    else:
        sampled, _ = gds.alpha.graph.sample.rwr(
            sample_name, gds.graph.get(graph_name), random_seed=seed
        )
    print(
        f"Sampled graph '{sample_name}' with {sampled.node_count()} nodes and {sampled.relationship_count()} edges."
    )
    return sampled


def fetch_topology(
    gds: GraphDataScience,
    graph,
) -> torch.LongTensor:
    """Fetch and normalize edge indices from sampled graph."""
    rel_df = gds.beta.graph.relationships.stream(graph)
    # Group by relationship type
    by_type = rel_df.by_rel_type()
    # Assuming single relation type 'IS_SIMILAR_TO'
    src, dst = by_type[next(iter(by_type))]
    # Obtain nodeId index mapping to consecutive IDs
    # Fetch node properties to get consistent nodeId ordering
    node_df = gds.graph.nodeProperties.stream(
        graph, ["embedding"], separate_property_columns=True
    )
    old_to_new = {old: new for new, old in enumerate(node_df["nodeId"])}
    src_idx = [old_to_new[n] for n in src]
    dst_idx = [old_to_new[n] for n in dst]
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    print(f"Constructed edge_index tensor with shape {edge_index.shape}.")
    return edge_index, node_df


def fetch_node_features(node_df: pd.DataFrame) -> torch.FloatTensor:
    """Convert node embeddings from DataFrame to tensor."""
    x = torch.tensor(node_df["embedding"].tolist(), dtype=torch.float)
    print(f"Loaded node feature matrix x with shape {x.shape}.")
    return x


def create_gds_graph(
    gds: GraphDataScience,
    graph_name: str,
) -> GraphDataScience:
    """Create a GDS graph from the Neo4j database."""
    if gds.graph.exists(graph_name)["exists"]:
        print(f"Graph '{graph_name}' already exists. Fetch it.")
        return gds.graph.get(graph_name)
    else:
        gds.run_cypher(
            f"""MATCH (source:CONTEXT)
		OPTIONAL MATCH (source:CONTEXT)-[r:IS_SIMILAR_TO]->(target:CONTEXT)
		RETURN gds.graph.project(
		  '{graph_name}',
		  source,
		  target,
		  {{
		    sourceNodeLabels: labels(source),
		    targetNodeLabels: labels(target),
		    sourceNodeProperties: source {{ .embedding }},
		    targetNodeProperties: target {{ .embedding }},
		    relationshipType: type(r),
		    relationshipProperties: r {{ .score }}
		  }},
		  {{ undirectedRelationshipTypes: ['IS_SIMILAR_TO'] }}
        )""",
        )
        print(f"Created graph '{graph_name}'.")
        return gds.graph.get(graph_name)
