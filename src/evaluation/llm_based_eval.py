import os
from typing import Any

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextRecall,
    FactualCorrectness,
    LLMContextPrecisionWithReference,
    ResponseRelevancy,
)

from utils.utils import save_json_file


class LLMBasedEvaluator:
    def __init__(self, llm: Any, embedding_model: Any):
        self.evaluator_llm = LangchainLLMWrapper(llm)
        self.evaluator_embedding = LangchainEmbeddingsWrapper(embedding_model)
        context_precision = LLMContextPrecisionWithReference()
        context_recall = ContextRecall()
        response_relevancy = ResponseRelevancy()
        factual_correctness = FactualCorrectness()
        self.eval_metrics = [
            context_precision,
            context_recall,
            response_relevancy,
            factual_correctness,
        ]

    def evaluate_answers(self, generated_data: list) -> dict:
        evaluation_dataset = EvaluationDataset.from_list(generated_data)
        metrics = evaluate(
            dataset=evaluation_dataset,
            metrics=self.eval_metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embedding,
        )
        return metrics


def run_evaluation_on_generated_answers(
    generated_data: list, llm: Any, embedding_model: Any, output_dir: str = None
) -> dict:
    evaluator = LLMBasedEvaluator(llm=llm, embedding_model=embedding_model)
    metrics = evaluator.evaluate_answers(generated_data=generated_data)

    if output_dir:
        # save metrics json
        metrics_file_name = "llm_based_metrics.json"
        save_json_file(
            file_path=os.path.join(output_dir, metrics_file_name), data=metrics
        )

    return metrics
