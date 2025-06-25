import pandas as pd
import torch
from graphdatascience import GraphDataScience
from typing import Dict, Tuple, List

# ----------------------------------
# Heterogeneous Data Extraction
# ----------------------------------


def connect_to_neo4j(
    uri: str, user: str, password: str, database: str
) -> GraphDataScience:
    """Create a GDS client connected to the specified Neo4j database."""
    return GraphDataScience(uri, auth=(user, password), database=database)


def fetch_heterogeneous_nodes(gds: GraphDataScience) -> Dict[str, pd.DataFrame]:
    """Fetch all node types (QA_PAIR, CONTEXT, MESH) with their properties."""
    
    # Fetch QA_PAIR nodes
    qa_query = """
    MATCH (qa:QA_PAIR)
    RETURN id(qa) as node_id, qa.id as qa_id, qa.question as question, 
           qa.answer as answer, qa.embedding as embedding
    """
    qa_df = gds.run_cypher(qa_query)
    print(f"Fetched {len(qa_df)} QA_PAIR nodes")
    
    # Fetch CONTEXT nodes
    context_query = """
    MATCH (ctx:CONTEXT)
    RETURN id(ctx) as node_id, ctx.pmid as pmid, ctx.title as title, 
           ctx.text_content as text_content, ctx.embedding as embedding
    """
    context_df = gds.run_cypher(context_query)
    print(f"Fetched {len(context_df)} CONTEXT nodes")
    
    # Fetch MESH nodes
    mesh_query = """
    MATCH (mesh:MESH)
    RETURN id(mesh) as node_id, mesh.name as name, mesh.definition as definition, 
           mesh.embedding as embedding
    """
    mesh_df = gds.run_cypher(mesh_query)
    print(f"Fetched {len(mesh_df)} MESH nodes")
    
    return {
        'qa_pair': qa_df,
        'context': context_df,
        'mesh': mesh_df
    }


def fetch_heterogeneous_edges(gds: GraphDataScience) -> Dict[str, pd.DataFrame]:
    """Fetch all edge types with their properties."""
    
    # Fetch HAS_CONTEXT edges
    has_context_query = """
    MATCH (qa:QA_PAIR)-[:HAS_CONTEXT]->(ctx:CONTEXT)
    RETURN id(qa) as source_id, id(ctx) as target_id
    """
    has_context_df = gds.run_cypher(has_context_query)
    print(f"Fetched {len(has_context_df)} HAS_CONTEXT edges")
    
    # Fetch IS_SIMILAR_TO edges
    similar_query = """
    MATCH (ctx1:CONTEXT)-[r:IS_SIMILAR_TO]->(ctx2:CONTEXT)
    RETURN id(ctx1) as source_id, id(ctx2) as target_id, r.score as score
    """
    similar_df = gds.run_cypher(similar_query)
    print(f"Fetched {len(similar_df)} IS_SIMILAR_TO edges")
    
    # Fetch HAS_MESH_TERM edges
    has_mesh_query = """
    MATCH (ctx:CONTEXT)-[:HAS_MESH_TERM]->(mesh:MESH)
    RETURN id(ctx) as source_id, id(mesh) as target_id
    """
    has_mesh_df = gds.run_cypher(has_mesh_query)
    print(f"Fetched {len(has_mesh_df)} HAS_MESH_TERM edges")
    
    return {
        'has_context': has_context_df,
        'is_similar_to': similar_df,
        'has_mesh_term': has_mesh_df
    }


def create_node_mappings(node_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Create mappings from original node IDs to consecutive indices for each node type."""
    mappings = {}
    
    for node_type, df in node_dfs.items():
        old_to_new = {old_id: idx for idx, old_id in enumerate(df['node_id'])}
        new_to_old = {idx: old_id for old_id, idx in old_to_new.items()}
        mappings[node_type] = {
            'old_to_new': old_to_new,
            'new_to_old': new_to_old,
            'num_nodes': len(df)
        }
        print(f"Created mapping for {node_type}: {len(df)} nodes")
    
    return mappings


def convert_edge_indices(
    edge_dfs: Dict[str, pd.DataFrame], 
    node_mappings: Dict[str, Dict],
    edge_type_mappings: Dict[str, Tuple[str, str]]
) -> Dict[str, torch.LongTensor]:
    """Convert edge indices using node mappings."""
    edge_indices = {}
    
    for edge_type, df in edge_dfs.items():
        if len(df) == 0:
            print(f"Warning: No edges found for {edge_type}")
            continue
            
        source_type, target_type = edge_type_mappings[edge_type]
        
        # Map source and target node IDs
        source_mapping = node_mappings[source_type]['old_to_new']
        target_mapping = node_mappings[target_type]['old_to_new']
        
        source_indices = [source_mapping[old_id] for old_id in df['source_id']]
        target_indices = [target_mapping[old_id] for old_id in df['target_id']]
        
        edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
        edge_indices[edge_type] = edge_index
        
        print(f"Converted {edge_type} edges: {edge_index.shape}")
    
    return edge_indices


def extract_node_features(node_dfs: Dict[str, pd.DataFrame]) -> Dict[str, torch.FloatTensor]:
    """Extract node features from DataFrames."""
    node_features = {}
    
    for node_type, df in node_dfs.items():
        # Extract embeddings
        embeddings = df['embedding'].tolist()
        features = torch.tensor(embeddings, dtype=torch.float)
        node_features[node_type] = features
        
        print(f"Extracted {node_type} features: {features.shape}")
    
    return node_features


def extract_edge_attributes(
    edge_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, torch.FloatTensor]:
    """Extract edge attributes where available."""
    edge_attrs = {}
    
    for edge_type, df in edge_dfs.items():
        if 'score' in df.columns:
            scores = torch.tensor(df['score'].tolist(), dtype=torch.float)
            edge_attrs[edge_type] = scores
            print(f"Extracted {edge_type} edge attributes: {scores.shape}")
        else:
            print(f"No attributes for {edge_type} edges")
    
    return edge_attrs


def sample_heterogeneous_subgraph(
    gds: GraphDataScience,
    max_qa_pairs: int = 1000,
    max_contexts_per_qa: int = 10,
    max_mesh_per_context: int = 5,
    seed: int = 42
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Sample a heterogeneous subgraph for training."""
    
    # Sample QA pairs
    sample_qa_query = f"""
    MATCH (qa:QA_PAIR)
    WITH qa, rand() as r
    ORDER BY r
    LIMIT {max_qa_pairs}
    RETURN id(qa) as node_id, qa.id as qa_id, qa.question as question, 
           qa.answer as answer, qa.embedding as embedding
    """
    qa_df = gds.run_cypher(sample_qa_query)
    qa_ids = qa_df['node_id'].tolist()
    
    # Sample contexts connected to these QA pairs
    sample_context_query = f"""
    MATCH (qa:QA_PAIR)-[:HAS_CONTEXT]->(ctx:CONTEXT)
    WHERE id(qa) IN $qa_ids
    WITH qa, ctx, rand() as r
    ORDER BY id(qa), r
    WITH qa, collect(ctx)[0..{max_contexts_per_qa}] as contexts
    UNWIND contexts as ctx
    RETURN DISTINCT id(ctx) as node_id, ctx.pmid as pmid, ctx.title as title, 
           ctx.text_content as text_content, ctx.embedding as embedding
    """
    context_df = gds.run_cypher(sample_context_query, params={'qa_ids': qa_ids})
    context_ids = context_df['node_id'].tolist()
    
    # Sample MESH terms connected to these contexts
    sample_mesh_query = f"""
    MATCH (ctx:CONTEXT)-[:HAS_MESH_TERM]->(mesh:MESH)
    WHERE id(ctx) IN $context_ids
    WITH ctx, mesh, rand() as r
    ORDER BY id(ctx), r
    WITH ctx, collect(mesh)[0..{max_mesh_per_context}] as meshes
    UNWIND meshes as mesh
    RETURN DISTINCT id(mesh) as node_id, mesh.name as name, mesh.definition as definition, 
           mesh.embedding as embedding
    """
    mesh_df = gds.run_cypher(sample_mesh_query, params={'context_ids': context_ids})
    
    # Get all relevant edges
    edges = {}
    
    # HAS_CONTEXT edges
    has_context_query = """
    MATCH (qa:QA_PAIR)-[:HAS_CONTEXT]->(ctx:CONTEXT)
    WHERE id(qa) IN $qa_ids AND id(ctx) IN $context_ids
    RETURN id(qa) as source_id, id(ctx) as target_id
    """
    edges['has_context'] = gds.run_cypher(has_context_query, params={'qa_ids': qa_ids, 'context_ids': context_ids})
    
    # IS_SIMILAR_TO edges (between sampled contexts)
    similar_query = """
    MATCH (ctx1:CONTEXT)-[r:IS_SIMILAR_TO]->(ctx2:CONTEXT)
    WHERE id(ctx1) IN $context_ids AND id(ctx2) IN $context_ids
    RETURN id(ctx1) as source_id, id(ctx2) as target_id, r.score as score
    """
    edges['is_similar_to'] = gds.run_cypher(similar_query, params={'context_ids': context_ids})
    
    # HAS_MESH_TERM edges
    mesh_ids = mesh_df['node_id'].tolist()
    has_mesh_query = """
    MATCH (ctx:CONTEXT)-[:HAS_MESH_TERM]->(mesh:MESH)
    WHERE id(ctx) IN $context_ids AND id(mesh) IN $mesh_ids
    RETURN id(ctx) as source_id, id(mesh) as target_id
    """
    edges['has_mesh_term'] = gds.run_cypher(has_mesh_query, params={'context_ids': context_ids, 'mesh_ids': mesh_ids})
    
    nodes = {
        'qa_pair': qa_df,
        'context': context_df,
        'mesh': mesh_df
    }
    
    print(f"Sampled subgraph: {len(qa_df)} QA pairs, {len(context_df)} contexts, {len(mesh_df)} MESH terms")
    return nodes, edges


def build_heterogeneous_data(
    gds: GraphDataScience,
    sample_subgraph: bool = True,
    max_qa_pairs: int = 1000,
    **sample_kwargs
) -> Dict:
    """Build complete heterogeneous graph data."""
    
    # Define edge type mappings (source_type, target_type)
    edge_type_mappings = {
        'has_context': ('qa_pair', 'context'),
        'is_similar_to': ('context', 'context'),
        'has_mesh_term': ('context', 'mesh')
    }
    
    if sample_subgraph:
        print("Sampling heterogeneous subgraph...")
        node_dfs, edge_dfs = sample_heterogeneous_subgraph(
            gds, max_qa_pairs=max_qa_pairs, **sample_kwargs
        )
    else:
        print("Fetching full heterogeneous graph...")
        node_dfs = fetch_heterogeneous_nodes(gds)
        edge_dfs = fetch_heterogeneous_edges(gds)
    
    # Create mappings
    node_mappings = create_node_mappings(node_dfs)
    
    # Convert edges
    edge_indices = convert_edge_indices(edge_dfs, node_mappings, edge_type_mappings)
    
    # Extract features
    node_features = extract_node_features(node_dfs)
    edge_attrs = extract_edge_attributes(edge_dfs)
    
    return {
        'node_features': node_features,
        'edge_indices': edge_indices,
        'edge_attrs': edge_attrs,
        'node_mappings': node_mappings,
        'node_dfs': node_dfs,
        'edge_dfs': edge_dfs,
        'edge_type_mappings': edge_type_mappings
    } 