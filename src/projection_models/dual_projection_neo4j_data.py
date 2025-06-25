"""
Neo4j Data Handler for Dual Projection Model Training

This module handles loading and processing data from Neo4j database for dual projection
model training. It retrieves real question embeddings from QA_PAIR nodes and context 
embeddings from CONTEXT nodes (both semantic and graph embeddings).

Key Components:
1. Neo4jDataLoader: Loads training data from Neo4j
2. DualProjectionDataset: Dataset for training with Neo4j data
3. Data preprocessing utilities
4. Hard negative mining with Neo4j queries
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
import random
from collections import defaultdict

from knowledge_graph.connection import Neo4jConnection


class Neo4jDualProjectionDataLoader:
    """
    Data loader for dual projection training using Neo4j database.
    
    This class loads:
    - Question embeddings from QA_PAIR.embedding
    - Semantic context embeddings from CONTEXT.embedding  
    - Graph structure embeddings from CONTEXT.graph_embedding
    - Uses HAS_CONTEXT relationships to find positive pairs
    """
    
    def __init__(self, neo4j_connection: Neo4jConnection):
        self.neo4j_driver = neo4j_connection.get_driver()
        
    def load_qa_context_pairs(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Load question-context pairs from Neo4j database.
        
        Returns:
            List of dictionaries with question and context data
        """
        print("🔍 Loading QA-Context pairs from Neo4j...")
        
        # Query to get question-context pairs with embeddings
        cypher = """
        MATCH (qa:QA_PAIR)-[:HAS_CONTEXT]->(context:CONTEXT)
        WHERE qa.embedding IS NOT NULL 
          AND context.embedding IS NOT NULL
          AND context.graph_embedding IS NOT NULL
        RETURN 
            qa.id AS question_id,
            qa.question AS question_text,
            qa.embedding AS question_embedding,
            context.pmid AS context_pmid,
            context.content AS context_content,
            context.embedding AS context_semantic_embedding,
            context.graph_embedding AS context_graph_embedding
        """ + (f"LIMIT {limit}" if limit else "")
        
        with self.neo4j_driver.session() as session:
            result = session.run(cypher)
            pairs = []
            
            for record in tqdm(result, desc="Loading pairs"):
                pairs.append({
                    'question_id': record['question_id'],
                    'question_text': record['question_text'],
                    'question_embedding': record['question_embedding'],
                    'context_pmid': record['context_pmid'],
                    'context_content': record['context_content'],
                    'context_semantic_embedding': record['context_semantic_embedding'],
                    'context_graph_embedding': record['context_graph_embedding']
                })
        
        print(f"✅ Loaded {len(pairs)} QA-Context pairs")
        return pairs
    
    def get_all_contexts(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all context nodes with their embeddings for negative sampling.
        
        Returns:
            List of context dictionaries
        """
        print("🔍 Loading all contexts for negative sampling...")
        
        cypher = """
        MATCH (context:CONTEXT)
        WHERE context.embedding IS NOT NULL
          AND context.graph_embedding IS NOT NULL
        RETURN 
            context.pmid AS pmid,
            context.content AS content,
            context.embedding AS semantic_embedding,
            context.graph_embedding AS graph_embedding
        """ + (f"LIMIT {limit}" if limit else "")
        
        with self.neo4j_driver.session() as session:
            result = session.run(cypher)
            contexts = []
            
            for record in tqdm(result, desc="Loading contexts"):
                contexts.append({
                    'pmid': record['pmid'],
                    'content': record['content'],
                    'semantic_embedding': record['semantic_embedding'],
                    'graph_embedding': record['graph_embedding']
                })
        
        print(f"✅ Loaded {len(contexts)} contexts")
        return contexts
    
    def find_hard_negatives_neo4j(
        self, 
        question_embedding: List[float], 
        positive_pmids: List[str],
        num_negatives: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Find hard negative contexts using Neo4j vector similarity.
        
        Args:
            question_embedding: Question embedding vector
            positive_pmids: PMIDs of positive contexts to exclude
            num_negatives: Number of negatives to return
            similarity_threshold: Minimum similarity for hard negatives
        
        Returns:
            List of hard negative context dictionaries
        """
        # Convert positive PMIDs to parameter format
        positive_pmids_param = positive_pmids if positive_pmids else []
        
        # Query for hard negatives using vector similarity
        cypher = """
        WITH $question_emb AS q_emb, $positive_pmids AS pos_pmids
        MATCH (context:CONTEXT)
        WHERE context.embedding IS NOT NULL
          AND context.graph_embedding IS NOT NULL
          AND NOT context.pmid IN pos_pmids
        WITH context, 
             vector.similarity.cosine(q_emb, context.embedding) AS similarity
        WHERE similarity >= $similarity_threshold
        ORDER BY similarity DESC
        LIMIT $num_negatives
        RETURN 
            context.pmid AS pmid,
            context.content AS content,
            context.embedding AS semantic_embedding,
            context.graph_embedding AS graph_embedding,
            similarity
        """
        
        with self.neo4j_driver.session() as session:
            result = session.run(
                cypher,
                question_emb=question_embedding,
                positive_pmids=positive_pmids_param,
                similarity_threshold=similarity_threshold,
                num_negatives=num_negatives * 2  # Get extra candidates
            )
            
            hard_negatives = []
            for record in result:
                hard_negatives.append({
                    'pmid': record['pmid'],
                    'content': record['content'],
                    'semantic_embedding': record['semantic_embedding'],
                    'graph_embedding': record['graph_embedding'],
                    'similarity': record['similarity']
                })
        
        # Return top num_negatives
        return hard_negatives[:num_negatives]
    
    def create_training_data(
        self, 
        max_samples: Optional[int] = None,
        use_hard_negatives: bool = True,
        num_hard_negatives: int = 5,
        train_split: float = 0.8
    ) -> Tuple[Dict, Dict]:
        """
        Create training and validation datasets from Neo4j data.
        
        Args:
            max_samples: Maximum number of samples to use
            use_hard_negatives: Whether to include hard negatives
            num_hard_negatives: Number of hard negatives per sample
            train_split: Train/validation split ratio
        
        Returns:
            Tuple of (train_data, val_data) dictionaries
        """
        print("🔧 Creating training data from Neo4j...")
        
        # Load QA-Context pairs
        qa_context_pairs = self.load_qa_context_pairs(limit=max_samples)
        
        if not qa_context_pairs:
            raise ValueError("No QA-Context pairs found in Neo4j database")
        
        # Group by question to handle multiple contexts per question
        question_groups = defaultdict(list)
        for pair in qa_context_pairs:
            question_groups[pair['question_id']].append(pair)
        
        # Create training samples
        training_samples = []
        for question_id, contexts in tqdm(question_groups.items(), desc="Processing questions"):
            # Use the first context as the primary positive
            primary_context = contexts[0]
            
            sample = {
                'question_id': question_id,
                'question_text': primary_context['question_text'],
                'question_embedding': primary_context['question_embedding'],
                'positive_context': {
                    'pmid': primary_context['context_pmid'],
                    'content': primary_context['context_content'],
                    'semantic_embedding': primary_context['context_semantic_embedding'],
                    'graph_embedding': primary_context['context_graph_embedding']
                },
                'all_positive_pmids': [ctx['context_pmid'] for ctx in contexts]
            }
            
            # Add hard negatives if requested
            if use_hard_negatives:
                try:
                    hard_negatives = self.find_hard_negatives_neo4j(
                        question_embedding=primary_context['question_embedding'],
                        positive_pmids=sample['all_positive_pmids'],
                        num_negatives=num_hard_negatives
                    )
                    sample['hard_negatives'] = hard_negatives
                except Exception as e:
                    print(f"⚠️ Failed to get hard negatives for question {question_id}: {e}")
                    sample['hard_negatives'] = []
            
            training_samples.append(sample)
        
        # Train/validation split
        random.shuffle(training_samples)
        split_idx = int(len(training_samples) * train_split)
        
        train_samples = training_samples[:split_idx]
        val_samples = training_samples[split_idx:]
        
        train_data = {
            'samples': train_samples,
            'num_samples': len(train_samples)
        }
        
        val_data = {
            'samples': val_samples,
            'num_samples': len(val_samples)
        }
        
        print(f"✅ Created training data:")
        print(f"   Training samples: {len(train_samples)}")
        print(f"   Validation samples: {len(val_samples)}")
        print(f"   Hard negatives enabled: {use_hard_negatives}")
        
        return train_data, val_data


class Neo4jDualProjectionDataset(Dataset):
    """
    PyTorch Dataset for dual projection training with Neo4j data.
    """
    
    def __init__(self, training_data: Dict, use_hard_negatives: bool = True):
        self.samples = training_data['samples']
        self.use_hard_negatives = use_hard_negatives
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Base item with question and positive context
        item = {
            'question_id': sample['question_id'],
            'question_text': sample['question_text'],
            'question_embedding': torch.tensor(sample['question_embedding'], dtype=torch.float32),
            'positive_semantic_embedding': torch.tensor(
                sample['positive_context']['semantic_embedding'], dtype=torch.float32
            ),
            'positive_graph_embedding': torch.tensor(
                sample['positive_context']['graph_embedding'], dtype=torch.float32
            ),
            'positive_pmid': sample['positive_context']['pmid'],
            'positive_content': sample['positive_context']['content']
        }
        
        # Add hard negatives if available and requested
        if self.use_hard_negatives and 'hard_negatives' in sample and sample['hard_negatives']:
            hard_negatives = sample['hard_negatives']
            
            # Convert to tensors
            negative_semantic = [neg['semantic_embedding'] for neg in hard_negatives]
            negative_graph = [neg['graph_embedding'] for neg in hard_negatives]
            
            item['negative_semantic_embeddings'] = torch.tensor(negative_semantic, dtype=torch.float32)
            item['negative_graph_embeddings'] = torch.tensor(negative_graph, dtype=torch.float32)
            item['negative_pmids'] = [neg['pmid'] for neg in hard_negatives]
            item['negative_similarities'] = [neg.get('similarity', 0.0) for neg in hard_negatives]
        
        return item


def collate_neo4j_dual_projection_batch(batch):
    """
    Custom collate function for Neo4j dual projection batches.
    """
    collated = {
        'question_ids': [item['question_id'] for item in batch],
        'question_texts': [item['question_text'] for item in batch],
        'question_embeddings': torch.stack([item['question_embedding'] for item in batch]),
        'positive_semantic_embeddings': torch.stack([item['positive_semantic_embedding'] for item in batch]),
        'positive_graph_embeddings': torch.stack([item['positive_graph_embedding'] for item in batch]),
        'positive_pmids': [item['positive_pmid'] for item in batch],
        'positive_contents': [item['positive_content'] for item in batch]
    }
    
    # Handle hard negatives if present
    if 'negative_semantic_embeddings' in batch[0]:
        # Find max number of negatives
        max_negatives = max(len(item.get('negative_pmids', [])) for item in batch)
        
        if max_negatives > 0:
            negative_semantic_batch = []
            negative_graph_batch = []
            negative_pmids_batch = []
            
            for item in batch:
                if 'negative_semantic_embeddings' in item:
                    neg_sem = item['negative_semantic_embeddings']
                    neg_graph = item['negative_graph_embeddings']
                    neg_pmids = item['negative_pmids']
                    
                    # Pad if necessary
                    if len(neg_pmids) < max_negatives:
                        padding_needed = max_negatives - len(neg_pmids)
                        
                        # Pad embeddings with zeros
                        sem_pad = torch.zeros(padding_needed, neg_sem.shape[1])
                        graph_pad = torch.zeros(padding_needed, neg_graph.shape[1])
                        
                        neg_sem = torch.cat([neg_sem, sem_pad], dim=0)
                        neg_graph = torch.cat([neg_graph, graph_pad], dim=0)
                        
                        # Pad PMIDs with empty strings
                        neg_pmids = neg_pmids + [''] * padding_needed
                    
                    negative_semantic_batch.append(neg_sem)
                    negative_graph_batch.append(neg_graph)
                    negative_pmids_batch.append(neg_pmids)
                else:
                    # Create dummy negatives for this sample
                    dummy_sem = torch.zeros(max_negatives, collated['positive_semantic_embeddings'].shape[1])
                    dummy_graph = torch.zeros(max_negatives, collated['positive_graph_embeddings'].shape[1])
                    dummy_pmids = [''] * max_negatives
                    
                    negative_semantic_batch.append(dummy_sem)
                    negative_graph_batch.append(dummy_graph)
                    negative_pmids_batch.append(dummy_pmids)
            
            collated['negative_semantic_embeddings'] = torch.stack(negative_semantic_batch)
            collated['negative_graph_embeddings'] = torch.stack(negative_graph_batch)
            collated['negative_pmids'] = negative_pmids_batch
    
    return collated


def get_embedding_statistics(embeddings: List[List[float]]) -> Dict[str, float]:
    """
    Get statistics about embedding vectors.
    
    Args:
        embeddings: List of embedding vectors
    
    Returns:
        Dictionary with statistics
    """
    if not embeddings:
        return {}
    
    embeddings_array = np.array(embeddings)
    
    return {
        'count': len(embeddings),
        'dimension': embeddings_array.shape[1] if len(embeddings) > 0 else 0,
        'mean_norm': float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
        'std_norm': float(np.std(np.linalg.norm(embeddings_array, axis=1))),
        'mean_values': embeddings_array.mean(axis=0).mean(),
        'std_values': embeddings_array.std(axis=0).mean()
    }


def validate_neo4j_data(neo4j_connection: Neo4jConnection) -> Dict[str, Any]:
    """
    Validate that Neo4j contains the required data for dual projection training.
    
    Args:
        neo4j_connection: Neo4j connection instance
    
    Returns:
        Validation results dictionary
    """
    print("🔍 Validating Neo4j data for dual projection training...")
    
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    try:
        with neo4j_connection.get_driver().session() as session:
            # Check QA_PAIR nodes with embeddings
            qa_count_result = session.run("""
                MATCH (qa:QA_PAIR)
                WHERE qa.embedding IS NOT NULL
                RETURN count(qa) AS count
            """)
            qa_count = qa_count_result.single()['count']
            
            # Check CONTEXT nodes with both embeddings
            context_count_result = session.run("""
                MATCH (context:CONTEXT)
                WHERE context.embedding IS NOT NULL
                  AND context.graph_embedding IS NOT NULL
                RETURN count(context) AS count
            """)
            context_count = context_count_result.single()['count']
            
            # Check HAS_CONTEXT relationships
            relationship_count_result = session.run("""
                MATCH (qa:QA_PAIR)-[:HAS_CONTEXT]->(context:CONTEXT)
                WHERE qa.embedding IS NOT NULL
                  AND context.embedding IS NOT NULL
                  AND context.graph_embedding IS NOT NULL
                RETURN count(*) AS count
            """)
            relationship_count = relationship_count_result.single()['count']
            
            # Store statistics
            validation_results['statistics'] = {
                'qa_pairs_with_embeddings': qa_count,
                'contexts_with_both_embeddings': context_count,
                'valid_qa_context_relationships': relationship_count
            }
            
            # Validation checks
            if qa_count == 0:
                validation_results['errors'].append("No QA_PAIR nodes with embedding found")
                validation_results['valid'] = False
            
            if context_count == 0:
                validation_results['errors'].append("No CONTEXT nodes with both embedding and graph_embedding found")
                validation_results['valid'] = False
            
            if relationship_count == 0:
                validation_results['errors'].append("No valid HAS_CONTEXT relationships found")
                validation_results['valid'] = False
            
            # Warnings for low data
            if qa_count < 100:
                validation_results['warnings'].append(f"Low number of QA pairs with embeddings: {qa_count}")
            
            if context_count < 1000:
                validation_results['warnings'].append(f"Low number of contexts with embeddings: {context_count}")
            
            if relationship_count < 100:
                validation_results['warnings'].append(f"Low number of training pairs: {relationship_count}")
            
            print(f"✅ Validation completed:")
            print(f"   QA pairs with embeddings: {qa_count}")
            print(f"   Contexts with both embeddings: {context_count}")
            print(f"   Valid training relationships: {relationship_count}")
            
            if validation_results['warnings']:
                print("⚠️ Warnings:")
                for warning in validation_results['warnings']:
                    print(f"   - {warning}")
            
            if validation_results['errors']:
                print("❌ Errors:")
                for error in validation_results['errors']:
                    print(f"   - {error}")
    
    except Exception as e:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Database query error: {str(e)}")
        print(f"❌ Validation failed: {e}")
    
    return validation_results 