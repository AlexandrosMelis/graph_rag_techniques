from typing import Optional

import numpy as np
import pandas as pd


class BioASQDataReader:
    def __init__(
        self,
        samples_start: int = 0,
        samples_end: Optional[int] = None,
    ):
        """
        :param samples_start: zero‐based index of the first row to include
        :param samples_end: zero‐based index to stop before (None = all remaining rows)
        """
        self.splits = {
            "train": "question-answer-passages/train-00000-of-00001.parquet",
            "test": "question-answer-passages/test-00000-of-00001.parquet",
        }
        self.samples_start = samples_start
        self.samples_end = samples_end

    def read_parquet_file(self, file_path: str) -> list:
        # try local, else fall back to HF
        try:
            print(f"Reading {file_path!r}...")
            self.df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Error reading {file_path!r}: {e}")
            # print("Downloading the dataset from Hugging Face Datasets...")
            # self.df = pd.read_parquet(
            #     "hf://datasets/enelpol/rag-mini-bioasq/" + self.splits["train"]
            # )

        # slice between start/end
        if self.samples_end is not None:
            print(
                f"Selecting rows from {self.samples_start} to {self.samples_end} (exclusive)..."
            )
            self.df = self.df[self.samples_start : self.samples_end]
        else:
            print(f"Selecting rows from {self.samples_start} to end...")
            self.df = self.df[self.samples_start :]

        print(f"Data file loaded with shape: {self.df.shape}")

        # convert 'relevant_passage_ids' arrays to list[str]
        self.df["relevant_passage_ids"] = self.df["relevant_passage_ids"].apply(
            lambda cell: (
                cell.astype(str).tolist() if isinstance(cell, np.ndarray) else cell
            )
        )

        return self.df.to_dict(orient="records")

    def get_distinct_pmids(self) -> list:
        """
        Explodes the list of passage‐IDs and returns unique PMIDs as strings.
        """
        int_pmids = self.df["relevant_passage_ids"].explode().unique()
        return [str(p) for p in int_pmids]
