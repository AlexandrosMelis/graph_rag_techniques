"""
Batched PubMed access through NCBI Entrez: MEDLINE records and MeSH headings for a
list of PMIDs, with retries and an on-disk cache so interrupted runs resume.
"""

import json
import time
from collections.abc import Iterable
from pathlib import Path

from Bio import Entrez, Medline
from tqdm import tqdm

from graph_rag.config import ConfigEnv
from graph_rag.data.entities import clean_mesh_term


def parse_medline_record(record: dict) -> dict:
    return {
        "pmid": record.get("PMID"),
        "title": record.get("TI"),
        "abstract": record.get("AB"),
        "mesh_terms": [clean_mesh_term(t) for t in record.get("MH", [])],
    }


class PubMedClient:
    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
        batch_size: int = 200,
        max_retries: int = 4,
    ):
        if email is None:
            ConfigEnv.require("ENTREZ_EMAIL")
            email = ConfigEnv.ENTREZ_EMAIL
        Entrez.email = email
        Entrez.api_key = api_key or ConfigEnv.ENTREZ_API_KEY
        self.batch_size = batch_size
        self.max_retries = max_retries

    def _efetch(self, pmids: list[str]) -> list[dict]:
        for attempt in range(self.max_retries):
            try:
                handle = Entrez.efetch(
                    db="pubmed", id=",".join(pmids), rettype="medline", retmode="text"
                )
                try:
                    return [parse_medline_record(r) for r in Medline.parse(handle)]
                finally:
                    handle.close()
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
        return []

    def fetch_records(self, pmids: Iterable[str]) -> list[dict]:
        pmids = list(dict.fromkeys(str(p) for p in pmids))
        records = []
        for start in range(0, len(pmids), self.batch_size):
            records.extend(self._efetch(pmids[start : start + self.batch_size]))
        return records

    def fetch_mesh_headings(
        self, pmids: Iterable[str], cache_path: str | Path | None = None
    ) -> dict[str, list[str]]:
        """
        Return {pmid: [MeSH heading, ...]}. With `cache_path`, results are appended to a
        JSONL file after every batch and PMIDs already present are skipped.
        """
        headings: dict[str, list[str]] = {}
        if cache_path and Path(cache_path).exists():
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    headings[row["pmid"]] = row["mesh_terms"]

        todo = [p for p in dict.fromkeys(str(p) for p in pmids) if p not in headings]
        batches = range(0, len(todo), self.batch_size)
        for start in tqdm(batches, desc="Fetching MeSH headings", disable=not todo):
            batch = todo[start : start + self.batch_size]
            fetched = {r["pmid"]: r["mesh_terms"] for r in self._efetch(batch) if r["pmid"]}
            # PMIDs without a MEDLINE record still get an entry so they are not refetched.
            rows = [{"pmid": p, "mesh_terms": fetched.get(p, [])} for p in batch]
            headings.update({r["pmid"]: r["mesh_terms"] for r in rows})
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "a", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
        return headings
