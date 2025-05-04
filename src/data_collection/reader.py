import json
import os

import numpy as np
import pandas as pd

from configs.config import ConfigPath, logger


class PQADataReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data: dict = {}
        self.pmids: list = []
        self._read_file()

    def _read_file(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                self.data = json.load(file)
                if not self.data:
                    raise ValueError("Metadata json file cannot be empty!")
                self.pmids = list(self.data.keys())
                if not self.pmids:
                    raise ValueError("PMIDs not found!")
                logger.debug("External metadata file loaded successfully!")
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {self.file_path}")
        except ValueError as ve:
            logger.error(f"Value error: {ve}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

    def get_pmids(self) -> list:
        return self.pmids


class BioASQDataReader:

    def __init__(self, samples_limit: int = 1000):
        # os.path.join(ConfigPath.RAW_DATA_DIR,"train-00000-of-00001.parquet")
        self.splits = {
            "train": "question-answer-passages/train-00000-of-00001.parquet",
            "test": "question-answer-passages/test-00000-of-00001.parquet",
        }
        self.samples_limit = samples_limit

    def read_parquet_file(self, file_path: str) -> list:
        try:
            self.df = pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"Error reading parquet file: {e}")
            logger.debug("Downloading the dataset from Hugging Face Datasets...")
            self.df = pd.read_parquet(
                "hf://datasets/enelpol/rag-mini-bioasq/" + self.splits["train"]
            )

        logger.info(f"Limiting the number of rows to {self.samples_limit}...")
        self.df = self.df[: self.samples_limit]
        logger.info(f"Data file loaded with shape: {self.df.shape}")
        # convert column 'relevant_passage_ids' from array[int] -> list[str]
        self.df["relevant_passage_ids"] = self.df["relevant_passage_ids"].apply(
            lambda cell_value: (
                cell_value.astype(str).tolist()
                if isinstance(cell_value, np.ndarray)
                else cell_value
            )
        )
        records = self.df.to_dict(orient="records")
        return records

    def get_distinct_pmids(self) -> list:
        int_pmids = list(self.df["relevant_passage_ids"].explode().unique())
        pmids = [str(pmid) for pmid in int_pmids]
        return pmids
