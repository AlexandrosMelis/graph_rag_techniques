# from configs.config import ConfigEnv
# from graph_embeddings.projection_data_processor import DataProcessor
# from projection_models.graph_aware.train_gat_projection_model import train_gat_projection


# def train_model():
#     # 1) Fetch your DataFrame of (q_emb, c_emb) pairs
#     df = DataProcessor(
#         uri=ConfigEnv.NEO4J_URI,
#         user=ConfigEnv.NEO4J_USER,
#         password=ConfigEnv.NEO4J_PASSWORD,
#         database=ConfigEnv.NEO4J_DB,
#     ).fetch_pairs()

#     # 2) Train
#     neo4j_params = dict(
#         uri=ConfigEnv.NEO4J_URI,
#         user=ConfigEnv.NEO4J_USER,
#         password=ConfigEnv.NEO4J_PASSWORD,
#         database=ConfigEnv.NEO4J_DB,
#     )
#     model, history = train_gat_projection(
#         df,
#         neo4j_params,
#         in_dim=768,
#         hidden_dim=512,
#         out_dim=128,
#         lr=1e-3,
#         weight_decay=1e-4,
#         batch_size=1,
#         epochs=200,
#         device="cuda",
#     )


# def project():
#     import torch

#     # 3) Project a new query
#     from projection_models.graph_aware.query_gat_loader import QueryGATLoader
#     from llms.embedding_model import EmbeddingModel

#     embedding_model = EmbeddingModel()

#     loader = QueryGATLoader(**neo4j_params)
#     q_emb = embedding_model.embed_documents(["Your new question"])[0]
#     data = loader.build_subgraph(torch.tensor(q_emb), top_k=10).to("cuda")
#     proj_emb = model(data.x, data.edge_index)  # torch.Tensor[128]


# if __name__ == "__main__":
#     train_model()
