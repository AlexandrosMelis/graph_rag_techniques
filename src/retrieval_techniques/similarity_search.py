from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate

from knowledge_graph.connection import Neo4jConnection


class SimilaritySearchRetriever:

    # available indexes
    context_vector_index = "contextIndex"
    mesh_vector_index = "meshIndex"

    # cypher snippets
    CONTEXT_VECTOR_SEARCH_CYPHER = """CALL db.index.vector.queryNodes('contextIndex', $k, $embedded_query) YIELD node AS context, score""".strip()
    MESH_VECTOR_SEARCH_CYPHER = """CALL db.index.vector.queryNodes('meshIndex', $k, $embedded_query) YIELD node AS mesh, score""".strip()
    CONTEXT_RETRIEVAL_CYPHER_SNIPPET = """MATCH (article:ARTICLE)-[:HAS_CONTEXT]->(context) RETURN elementId(context) as element_id, article.pmid as pmid, context.text_content as content, score as score""".strip()
    MESH_RETRIEVAL_CYPHER_SNIPPET = """RETURN mesh.name as term, mesh.definition as definition, score as score""".strip()
    SIMILAR_CONTEXT_RETRIEVAL_CYPHER_SNIPPET = """MATCH (article:ARTICLE)-[:HAS_CONTEXT]->(context:CONTEXT) WHERE elementId(context) in $relevant_element_ids
        // For each context, find its top 5 most similar contexts
        WITH context, article.pmid AS original_pmid, context.text_content AS original_content
        MATCH (context)-[sim:IS_SIMILAR_TO]-(similar_context:CONTEXT)<-[:HAS_CONTEXT]-(similar_article:ARTICLE)
        // Group by original context and order similar contexts by similarity score
        WITH context, original_pmid, original_content, 
            elementId(context) AS original_context_element_id,
            similar_context.text_content AS similar_content, similar_article.pmid AS similar_pmid,
            sim.score AS similarity_score, elementId(similar_context) AS similar_element_id
        ORDER BY similarity_score DESC
        // Collect top n similar contexts for each original context
        WITH original_context_element_id, original_pmid,
            collect({
                element_id: similar_element_id,
                pmid: similar_pmid, 
                content: similar_content,
                score: similarity_score
            })[0..$n] AS top_similar_contexts
        // Return the results
        RETURN original_context_element_id AS element_id, original_pmid AS pmid, top_similar_contexts""".strip()

    def __init__(
        self,
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        neo4j_connection: Neo4jConnection,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.neo4j_connection = neo4j_connection
        self.str_parser = StrOutputParser()

        # retrieval techniques mapper
        self.retrieval_techniques_mapper = {
            "context_vector_search": self.get_relevant_contexts,
            "1_hop_context_expansion": self.get_1_hop_similar_contexts,
            "mesh_centrality_contexts": self.get_mesh_centrality_contexts,
        }

        # answer techniques mapper
        self.answer_techniques_mapper = {
            "relevant_contexts": self.get_answer_based_on_contexts,
        }

    def perform_retrieval(self, retrieval_type: str, query: str, k: int, **kwargs):
        """
        Factory method to retrieve chunks based on the technique specified.
        Entry point for the chunk retrieval process.
        """
        if retrieval_type not in self.retrieval_techniques_mapper:
            raise ValueError(
                f"Retrieval technique '{retrieval_type}' is not supported. Valid options are: {list(self.retrieval_techniques_mapper.keys())}."
            )
        return self.retrieval_techniques_mapper[retrieval_type](
            query=query, k=k, **kwargs
        )

    def search_in_contexts(self, query: str, k: int, retrieval_snippet: str) -> list:
        """
        Perform similarity search in Context index and retrieve the `k` most relevant chunks.
        """
        CYPHER_QUERY = (
            f"{self.CONTEXT_VECTOR_SEARCH_CYPHER}\n{retrieval_snippet}".strip()
        )
        embedded_query = self.embedding_model.embed_query(query)
        chunks = self.neo4j_connection.execute_query(
            query=CYPHER_QUERY,
            params={
                "embedded_query": embedded_query,
                "k": k,
            },
        )

        return chunks

    def search_in_meshes(self, query: str, k: int, retrieval_snippet: str) -> list:
        """
        Perform similarity search in Mesh index and retrieve the `k` most relevant chunks.
        """
        CYPHER_QUERY = f"{self.MESH_VECTOR_SEARCH_CYPHER}\n{retrieval_snippet}".strip()

        embedded_query = self.embedding_model.embed_query(query)
        chunks = self.neo4j_connection.execute_query(
            query=CYPHER_QUERY,
            params={
                "embedded_query": embedded_query,
                "k": k,
            },
        )

        return chunks

    def get_relevant_meshes(self, query: str, k: int) -> list:
        """
        Get `k` most relevant Mesh terms to the query.
        """
        return self.search_in_meshes(
            query=query, k=k, retrieval_snippet=self.MESH_RETRIEVAL_CYPHER_SNIPPET
        )

    def get_relevant_contexts(self, query: str, k: int) -> list:
        """
        Get `k` most relevant contexts to the query.
        """
        return self.search_in_contexts(
            query=query,
            k=k,
            retrieval_snippet=self.CONTEXT_RETRIEVAL_CYPHER_SNIPPET,
        )

    def perform_meshes_subgraph_search(
        self, query: str, k: int, n_meshes: int = 10
    ) -> list:

        # embed query
        embedded_query = self.embedding_model.embed_query(query)

        # get relevant meshes
        relevant_meshes = self.get_relevant_meshes(query=query, k=n_meshes)
        relevant_mesh_terms = [mesh["term"] for mesh in relevant_meshes]

        # apply graph prefiltering
        PREFILTERING_VECTOR_SEARCH_CYPHER = """MATCH (article:ARTICLE)-[:HAS_CONTEXT]->(context:CONTEXT)-[:HAS_MESH_TERM]->(mesh:MESH)
        WHERE mesh.name IN $mesh_terms
        WITH article, context, mesh, vector.similarity.cosine(context.embedding, $embedding) AS score 
        ORDER BY score DESC LIMIT toInteger($k) 
        RETURN article.pmid as pmid, context.text_content as content, score as score""".strip()

        chunks = self.neo4j_connection.execute_query(
            query=PREFILTERING_VECTOR_SEARCH_CYPHER,
            params={
                "embedding": embedded_query,
                "k": k,
                "mesh_terms": relevant_mesh_terms,
            },
        )

        return chunks

    def perform_enhanced_mesh_search(
        self,
        query: str,
        k: int,
        n_meshes: int = 10,
        mesh_weight: float = 0.3,
        expansion_factor: float = 0.5,
    ) -> list:
        """
        Enhanced mesh-based retrieval that uses mesh term relevance to guide the context search.

        Args:
            query: The user query
            k: Number of contexts to retrieve
            n_meshes: Number of primary mesh terms to consider
            mesh_weight: Weight given to mesh term relevance (0-1)
            expansion_factor: Factor for including related mesh terms (0-1)

        Returns:
            List of retrieved contexts with combined scoring
        """
        # Embed query
        embedded_query = self.embedding_model.embed_query(query)

        # 1. Get primary relevant mesh terms with scores
        primary_meshes = self.get_relevant_meshes(query=query, k=n_meshes)
        primary_mesh_terms = [mesh["term"] for mesh in primary_meshes]

        # Create a dictionary to store mesh term scores
        mesh_scores = {mesh["term"]: mesh["score"] for mesh in primary_meshes}

        # 2. Expand to related mesh terms (those frequently co-occurring with primary terms)
        n_expanded = int(n_meshes * expansion_factor)
        if n_expanded > 0:
            MESH_EXPANSION_CYPHER = """
            MATCH (mesh:MESH)<-[:HAS_MESH_TERM]-(context:CONTEXT)-[:HAS_MESH_TERM]->(related_mesh:MESH)
            WHERE mesh.name IN $primary_mesh_terms
            AND NOT related_mesh.name IN $primary_mesh_terms
            WITH related_mesh, COUNT(DISTINCT context) AS co_occurrence_count
            ORDER BY co_occurrence_count DESC
            LIMIT $n_expanded
            RETURN related_mesh.name AS term
            """

            expanded_meshes = self.neo4j_connection.execute_query(
                query=MESH_EXPANSION_CYPHER,
                params={
                    "primary_mesh_terms": primary_mesh_terms,
                    "n_expanded": n_expanded,
                },
            )

            expanded_mesh_terms = [mesh["term"] for mesh in expanded_meshes]

            # Assign a lower score to expanded terms (half of the lowest primary term score)
            min_primary_score = min(mesh_scores.values()) if mesh_scores else 0.5
            for term in expanded_mesh_terms:
                mesh_scores[term] = min_primary_score * 0.5
        else:
            expanded_mesh_terms = []

        # Combine primary and expanded mesh terms
        all_mesh_terms = primary_mesh_terms + expanded_mesh_terms

        # 3. Retrieve contexts using a hybrid approach that considers both:
        #    - Mesh term relevance (weighted by term scores)
        #    - Direct vector similarity to the query
        HYBRID_SEARCH_CYPHER = """
        // First, find contexts connected to relevant mesh terms
        MATCH (article:ARTICLE)-[:HAS_CONTEXT]->(context:CONTEXT)-[:HAS_MESH_TERM]->(mesh:MESH)
        WHERE mesh.name IN $mesh_terms
        
        // Calculate mesh relevance score - sum of the scores of all matching mesh terms
        WITH article, context, 
             SUM(
                CASE 
                    WHEN mesh.name IN $mesh_terms THEN $mesh_scores[mesh.name] 
                    ELSE 0 
                END
             ) AS mesh_relevance
        
        // Calculate direct vector similarity between context and query
        WITH article, context, mesh_relevance,
             vector.similarity.cosine(context.embedding, $embedding) AS vector_similarity
        
        // Combine scores with weighting
        WITH article, context,
             (1.0 - $mesh_weight) * vector_similarity + $mesh_weight * mesh_relevance AS combined_score
        
        // Order by combined score and take top k
        ORDER BY combined_score DESC
        LIMIT toInteger($k)
        
        // Return the results
        RETURN 
            article.pmid AS pmid, 
            context.text_content AS content, 
            combined_score AS score
        """

        # Convert mesh_scores to a format that can be used in Cypher
        mesh_scores_param = {}
        for term, score in mesh_scores.items():
            mesh_scores_param[term] = float(score)

        # Execute hybrid search
        chunks = self.neo4j_connection.execute_query(
            query=HYBRID_SEARCH_CYPHER,
            params={
                "embedding": embedded_query,
                "k": k,
                "mesh_terms": all_mesh_terms,
                "mesh_scores": mesh_scores_param,
                "mesh_weight": mesh_weight,
            },
        )

        # 4. Handle the case where mesh filtering is too restrictive
        # If we didn't get enough results, perform direct vector search as a fallback
        if len(chunks) < k:
            # Calculate how many more results we need
            remaining_k = k - len(chunks)

            # Get element IDs of already retrieved contexts to exclude them
            retrieved_pmids = [chunk["pmid"] for chunk in chunks]

            FALLBACK_VECTOR_SEARCH_CYPHER = """
            MATCH (article:ARTICLE)-[:HAS_CONTEXT]->(context:CONTEXT)
            WHERE NOT article.pmid IN $excluded_pmids
            WITH article, context, vector.similarity.cosine(context.embedding, $embedding) AS score
            ORDER BY score DESC
            LIMIT toInteger($remaining_k)
            RETURN article.pmid AS pmid, context.text_content AS content, score AS score
            """

            fallback_chunks = self.neo4j_connection.execute_query(
                query=FALLBACK_VECTOR_SEARCH_CYPHER,
                params={
                    "embedding": embedded_query,
                    "remaining_k": remaining_k,
                    "excluded_pmids": retrieved_pmids,
                },
            )

            # Add fallback results to our chunks
            chunks.extend(fallback_chunks)

        return chunks

    def get_1_hop_similar_contexts(
        self, query: str, k: int, n_similar_contexts: int
    ) -> list:
        """ """
        # retrieve top k relevant contexts
        relevant_contexts = self.get_relevant_contexts(query=query, k=k)
        # get the context element ids
        relevant_element_ids = [context["element_id"] for context in relevant_contexts]

        embedded_query = self.embedding_model.embed_query(query)

        # expand to 1-hop similar context and get top n similar neighbors for each context
        retrieved_similar_contexts = self.neo4j_connection.execute_query(
            query=self.SIMILAR_CONTEXT_RETRIEVAL_CYPHER_SNIPPET,
            params={
                "embedded_query": embedded_query,
                "n": n_similar_contexts,
                "relevant_element_ids": relevant_element_ids,
            },
        )

        # format results to appropriate format to concatenate two result lists
        formatted_similar_contexts = list(
            {
                similar_context["element_id"]: {
                    "element_id": similar_context["element_id"],
                    "pmid": similar_context["pmid"],
                    "content": similar_context["content"],
                    "score": similar_context["score"],
                }
                for retrieved_similar_context in retrieved_similar_contexts
                for similar_context in retrieved_similar_context["top_similar_contexts"]
            }.values()
        )

        # get distinct results based on `element_id`
        seen = set()
        results = [
            item
            for item in relevant_contexts + formatted_similar_contexts
            if item.get("element_id") not in seen
            and not seen.add(item.get("element_id"))
        ]

        return results

    def get_mesh_centrality_contexts(
        self,
        query: str,
        k: int,
        centrality_type: str = "degree",
        centrality_weight: float = 0.3,
    ) -> list:
        """
        Retrieve contexts based on both vector similarity and the centrality of connected MESH terms.

        This method enhances retrieval by considering not just semantic relevance (vector similarity)
        but also the structural importance of medical concepts (MESH terms) in the knowledge graph.

        Args:
            query: The user query
            k: Number of contexts to retrieve
            centrality_type: Type of centrality to use ("degree", "betweenness", or "eigenvector")
            centrality_weight: Weight to apply to centrality scores (0-1)

        Returns:
            List of retrieved contexts ordered by combined score
        """
        print(f"Centrality type: {centrality_type}")

        # Step 1: Get initial contexts based on vector similarity
        initial_contexts = self.get_relevant_contexts(
            query=query, k=k * 2
        )  # Get more than needed initially

        # Extract the element IDs of the retrieved contexts
        context_element_ids = [context["element_id"] for context in initial_contexts]

        # Step 2: Calculate MESH term centrality based on the specified type
        if centrality_type == "degree":
            # Degree centrality: how many contexts are connected to each MESH term
            MESH_CENTRALITY_CYPHER = """
            MATCH (context:CONTEXT)-[:HAS_MESH_TERM]->(mesh:MESH)
            WHERE elementId(context) IN $context_element_ids
            
            WITH mesh, COUNT(DISTINCT context) AS degree_centrality
            
            MATCH (related_context:CONTEXT)-[:HAS_MESH_TERM]->(mesh)
            WHERE elementId(related_context) IN $context_element_ids
            
            RETURN 
                elementId(related_context) AS element_id, 
                mesh.name AS mesh_term,
                degree_centrality AS centrality
            """
        elif centrality_type == "betweenness":
            # Betweenness centrality approximation: how often a MESH term connects different contexts
            MESH_CENTRALITY_CYPHER = """
            MATCH (context:CONTEXT)-[:HAS_MESH_TERM]->(mesh:MESH)
            WHERE elementId(context) IN $context_element_ids
            
            WITH mesh
            MATCH (mesh)<-[:HAS_MESH_TERM]-(c:CONTEXT)-[:HAS_MESH_TERM]->(other_mesh:MESH)
            WHERE mesh <> other_mesh
            
            WITH mesh, COUNT(DISTINCT other_mesh) AS betweenness
            
            MATCH (related_context:CONTEXT)-[:HAS_MESH_TERM]->(mesh)
            WHERE elementId(related_context) IN $context_element_ids
            
            RETURN 
                elementId(related_context) AS element_id, 
                mesh.name AS mesh_term,
                betweenness AS centrality
            """
        elif centrality_type == "eigenvector":
            # Eigenvector centrality approximation: importance based on connections to other important MESH terms
            MESH_CENTRALITY_CYPHER = """
            MATCH (context:CONTEXT)-[:HAS_MESH_TERM]->(mesh:MESH)
            WHERE elementId(context) IN $context_element_ids
            
            WITH mesh
            MATCH (mesh)<-[:HAS_MESH_TERM]-(c:CONTEXT)-[:HAS_MESH_TERM]->(connected_mesh:MESH)<-[:HAS_MESH_TERM]-(other_c:CONTEXT)
            WHERE mesh <> connected_mesh
            
            WITH mesh, COUNT(DISTINCT other_c) AS eigenvector_approx
            
            MATCH (related_context:CONTEXT)-[:HAS_MESH_TERM]->(mesh)
            WHERE elementId(related_context) IN $context_element_ids
            
            RETURN 
                elementId(related_context) AS element_id, 
                mesh.name AS mesh_term,
                eigenvector_approx AS centrality
            """
        else:
            raise ValueError(
                f"Centrality type '{centrality_type}' is not supported. Valid options are: 'degree', 'betweenness', 'eigenvector'."
            )

        # Execute the centrality query
        mesh_centrality_results = self.neo4j_connection.execute_query(
            query=MESH_CENTRALITY_CYPHER,
            params={"context_element_ids": context_element_ids},
        )

        # Step 3: Aggregate MESH centrality scores by context
        context_to_centrality = {}
        for result in mesh_centrality_results:
            element_id = result["element_id"]
            centrality = result["centrality"]

            if element_id not in context_to_centrality:
                context_to_centrality[element_id] = []

            context_to_centrality[element_id].append(centrality)

        # Calculate average centrality for each context
        context_avg_centrality = {}
        for element_id, centralities in context_to_centrality.items():
            context_avg_centrality[element_id] = sum(centralities) / len(centralities)

        # Normalize centrality scores to 0-1 range if there are any scores
        if context_avg_centrality:
            max_centrality = max(context_avg_centrality.values())
            if max_centrality > 0:  # Avoid division by zero
                for element_id in context_avg_centrality:
                    context_avg_centrality[element_id] /= max_centrality

        # Step 4: Combine vector similarity and centrality scores
        combined_results = []
        for context in initial_contexts:
            element_id = context["element_id"]
            similarity_score = context["score"]
            centrality_score = context_avg_centrality.get(element_id, 0)

            # Calculate combined score
            combined_score = (
                1 - centrality_weight
            ) * similarity_score + centrality_weight * centrality_score

            combined_results.append(
                {
                    "element_id": element_id,
                    "pmid": context["pmid"],
                    "content": context["content"],
                    "similarity_score": similarity_score,
                    "centrality_score": centrality_score,
                    "combined_score": combined_score,
                }
            )

        # Sort by combined score and take top k
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        top_k_results = combined_results[:k]

        # Format results to match existing output structure
        formatted_results = [
            {
                "element_id": r["element_id"],
                "pmid": r["pmid"],
                "content": r["content"],
                "score": r["combined_score"],
            }
            for r in top_k_results
        ]

        return formatted_results

    ####################
    # ANSWER GENERATION #
    ####################

    def answer(self, **kwargs):
        """
        Factory method to get the answer based on the technique specified.
        Entry point for the answer retrieval process.
        """
        technique = kwargs.get("technique", None)
        if technique is None:
            raise ValueError("Technique is not defined. `technique` must be provided.")
        if technique not in self.answer_techniques_mapper:
            raise ValueError(
                f"Answer technique '{technique}' is not supported. Valid options are: {list(self.answer_techniques_mapper.keys())}."
            )
        if "k" not in kwargs:
            raise ValueError("k neighbors is not defined. `k` must be provided.")
        if "query" not in kwargs:
            raise ValueError("Query is not defined. `query` must be provided.")
        return self.answer_techniques_mapper[technique](
            query=kwargs["query"], k=kwargs["k"]
        )

    def _get_answer_template(self) -> PromptTemplate:
        LONG_ANSWER_PROMPT_TEMPLATE = """Role: You are a research expert in medical literature. 
<task>
Given the following context retrieved from PubMed article abstracts, answer the question.
Find the context placed in <context></context> tags and the question placed in <question></question> tags.  
</task>      

<instructions>
- Your answer should be based ONLY on the information presented in the context.
- Include the reasoning behind your answer and the conclusion.
- If there is no sufficient information in the context to answer the question, respond with "Cannot answer based on the provided information".
</instructions>

<output_format>
- Your output should be a consistent paragraph.
</output_format>

--Real data--
<context>
{context}
</context>

<question>
{question}
</question>""".strip()
        prompt_template = PromptTemplate.from_template(LONG_ANSWER_PROMPT_TEMPLATE)
        return prompt_template

    def get_answer_based_on_contexts(self, query: str, k: int) -> dict:
        """
        Get the answer based on the retrieved contexts using the LLM.
        """
        # get the relevant contexts
        relevant_contexts = self.get_relevant_contexts(query=query, k=k)
        contexts_str = "- " + "\n- ".join(
            [chunk["content"] for chunk in relevant_contexts]
        )

        prompt = self._get_answer_template()
        answer_chain = prompt | self.llm | self.str_parser

        answer = answer_chain.invoke({"question": query, "context": contexts_str})
        return {"answer": answer, "context": relevant_contexts}
