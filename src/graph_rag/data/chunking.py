from typing import Optional

import pandas as pd
from langchain_text_splitters import TokenTextSplitter


class TextSplitter:
    def __init__(self, chunk_size: Optional[int] = 512, chunk_overlap: Optional[int] = 100):
        self.encoding_name = "cl100k_base"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = TokenTextSplitter(
            encoding_name=self.encoding_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split_text(self, text: str):
        chunks = self.text_splitter.split_text(text)
        return chunks


def chunk_corpus(
    corpus: pd.DataFrame, chunk_size: int = 384, chunk_overlap: int = 64
) -> pd.DataFrame:
    """
    Split every passage into token windows. Returns one row per chunk with
    `chunk_id` (`<pmid>#<n>`), `pmid`, `chunk_index` and `text`; relevance is judged
    per PMID, so evaluation maps chunks back to their passage.
    """
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    rows = []
    for pmid, text in zip(corpus["pmid"], corpus["text"]):
        for i, chunk in enumerate(splitter.split_text(text) or [text]):
            rows.append({"chunk_id": f"{pmid}#{i}", "pmid": pmid, "chunk_index": i, "text": chunk})
    return pd.DataFrame(rows, columns=["chunk_id", "pmid", "chunk_index", "text"])
