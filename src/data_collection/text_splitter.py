from typing import Optional

from langchain_text_splitters import TokenTextSplitter


class TextSplitter:
    def __init__(
        self, chunk_size: Optional[int] = 512, chunk_overlap: Optional[int] = 100
    ):
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
