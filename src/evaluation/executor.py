import os
from typing import Any, Literal, Tuple

import pandas as pd
from tqdm import tqdm

from retrieval_techniques.base_retriever import BaseRetriever
from utils.utils import save_json_file


def run_retrieval(
    source_data: list,
    retriever: BaseRetriever,
    retriever_args: dict,
    output_dir: str,
) -> list:
    """
    Function responsible for running the provided retriever against the source data.
    It saves the retrieved contexts and their metadata for further evaluation.
    """
    results = []
    for example in tqdm(source_data, desc="Retrieving contexts..."):
        example_id = example.get("id")
        query = example.get("question")
        true_pmids = example.get("relevant_passage_ids")

        retrieved_contexts = retriever.retrieve(query=query, **retriever_args)
        retrieved_pmids = [context["pmid"] for context in retrieved_contexts]
        retrieved_scores = [context["score"] for context in retrieved_contexts]

        results.append(
            {
                "id": example_id,
                "query": query,
                "true_pmids": true_pmids,
                "retrieved_pmids": retrieved_pmids,
                "retrieved_scores": retrieved_scores,
            }
        )

    # save results to csv
    df = pd.DataFrame(results)
    file_name = f"{retriever.name}_retrieval_results.csv"
    df.to_csv(os.path.join(output_dir, file_name), index=False)

    return results


def collect_generated_answers(
    source_data: list, retriever: Any, output_dir: str = None
) -> list:
    results = []
    for sample in tqdm(source_data, desc="Collecting answers and chunks..."):
        sample_id = sample.get("id")
        user_input = sample.get("question")
        reference = sample.get("answer")
        output = retriever.invoke(user_input)
        if "answer" not in output:
            raise ValueError(
                "The retriever did not return an answer. Check retriever initialization!"
            )
        response = output["answer"]
        contexts = output["context"]
        retrieved_contexts = [context["content"] for context in contexts]

        result = {
            "id": sample_id,
            "user_input": user_input,
            "reference": reference,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        }
        results.append(result)

    if output_dir:
        file_name = "retrieved_answers.json"
        file_path = os.path.join(output_dir, file_name)
        save_json_file(file_path=file_path, data=results)

    return results
