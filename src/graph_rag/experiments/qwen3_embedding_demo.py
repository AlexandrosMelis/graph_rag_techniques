"""Sanity check for Qwen3-Embedding: query prompts and cosine similarity against two documents."""

from sentence_transformers import SentenceTransformer

QUERIES = ["What is the capital of China?", "Explain gravity"]
DOCUMENTS = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to "
    "physical objects and is responsible for the movement of planets around the sun.",
]


def main(model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> None:
    # flash_attention_2 with padding_side="left" is faster on supported GPUs:
    # SentenceTransformer(model_name, model_kwargs={"attn_implementation": "flash_attention_2",
    #   "device_map": "auto"}, tokenizer_kwargs={"padding_side": "left"})
    model = SentenceTransformer(model_name)
    # Queries use the model's "query" prompt; documents are embedded as-is.
    query_embeddings = model.encode(QUERIES, prompt_name="query")
    document_embeddings = model.encode(DOCUMENTS)
    print(model.similarity(query_embeddings, document_embeddings))


if __name__ == "__main__":
    main()
