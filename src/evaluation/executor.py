import os
from typing import Any, Literal, Tuple

import pandas as pd
from tqdm import tqdm

from configs.config import ConfigPath
from retrieval_techniques.similarity_search import SimilaritySearchRetriever
from utils.utils import save_json_file


def collect_retrieved_chunks(
    source_data: list,
    retriever: SimilaritySearchRetriever,
    func_args: dict,
    output_dir: str = None,
) -> dict:
    """Run the retriever against every sample in the source data.
    Prepares the results for the evaluation.

    Returns:
        dict: contains the id, question and the retrieved chunks
    """
    results = {}
    results_for_save = {}
    retrieval_type = func_args["retrieval_type"]
    print(f"Retrieval type: {retrieval_type}")
    for sample in tqdm(source_data, desc="Collecting retrieved chunks..."):
        sample_id = sample.get("id")
        func_args.update({"query": sample["question"]})
        retrieved_data = retriever.retrieve_chunks(**func_args)
        results[sample_id] = [
            (chunk["pmid"], chunk["score"]) for chunk in retrieved_data
        ]

        retrieved_pmids = [chunk["pmid"] for chunk in retrieved_data]
        results_for_save[sample_id] = retrieved_pmids

    if output_dir:
        # save results in csv
        pmid_results = list(results_for_save.values())

        len_of_retrieved_results = [len(pmids) for pmids in pmid_results]
        data = {
            "id": list(results.keys()),
            "retrieved_pmids": pmid_results,
            "num_of_chunks": len_of_retrieved_results,
        }
        # data = {"id": list(results.keys()), "retrieved_chunks}
        df = pd.DataFrame(data)
        file_name = f"{retrieval_type}_chunks.csv"
        file_path = os.path.join(output_dir, file_name)
        df.to_csv(file_path, index=False)

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
