import json
import os
import re

from Bio import Entrez
from tqdm import tqdm

from configs.config import ConfigEnv, ConfigPath, logger
from utils.utils import read_json_file, save_json_file


class PubMedArticleFetcher:

    def __init__(self):

        Entrez.email = ConfigEnv.ENTREZ_EMAIL
        self._db = "pubmed"
        self._rettype = "medline"
        self._retmode = "text"
        self._pmid_regex_pattern = r"PMID-\s*(\d+)"
        self._distinct_mesh_terms = set()

    def get_mesh_terms(
        self,
    ) -> list:
        return list(self._distinct_mesh_terms)

    def clean_mesh_term(self, term: str) -> str:
        """
        Clean a MeSH term by removing any unwanted characters.
        """
        term = term.split("/")[0]
        return term.replace("*", "").strip()

    def fetch_articles(self, pmids: list) -> dict:

        if not pmids:
            raise ValueError(
                "No PMIDs provided. Please provide at least one PMID to fetch articles."
            )

        # batch fetch articles from Entrez
        pmids_str = ", ".join(pmids)
        logger.info(f"Fetching articles for total PMIDs: {len(pmids)}")

        try:
            handle = Entrez.efetch(
                db=self._db, id=pmids_str, rettype=self._rettype, retmode=self._retmode
            )
        except Exception as e:
            logger.error(
                f"Failed to fetch articles from Entrez for PMIDs: {pmids_str}. Error: {e}"
            )
            raise

        # articles fetched as a single string
        articles_str = handle.read()
        handle.close()

        # split articles into individual articles
        articles = articles_str.strip().split("\n\n")
        logger.info(f"Total articles fetched: {len(articles)}")

        # extract Pubmed data from articles
        pubmed_data = self.extract_data_from_articles(articles=articles)
        save_json_file(
            file_path=os.path.join(
                ConfigPath.RAW_DATA_DIR, "bioasq_pubmed_articles.json"
            ),
            data=pubmed_data,
        )
        logger.info("Articles saved successfully.")
        return pubmed_data

    def extract_pmid(self, text):
        match = re.search(self._pmid_regex_pattern, text)
        if match:
            return match.group(1)
        else:
            return None

    def extract_title(self, text):
        """
        Extract title from PubMed text.
        Title follows "TI - " tag and can span multiple lines until the next tag.
        """
        # Find the title section
        title_match = re.search(
            r"TI\s+-\s+(.*?)(?=\n[A-Z]{2,4}\s+-\s+|\Z)", text, re.DOTALL
        )

        if not title_match:
            return None

        # Get the title text and clean it
        title_text = title_match.group(1).strip()

        # Join multiple lines and remove excess whitespace
        title_text = " ".join([line.strip() for line in title_text.split("\n")])

        return title_text

    def extract_abstract(self, text):
        """
        Extract abstract from PubMed text.
        Abstract follows "AB - " tag and can span multiple lines until the next tag.
        """
        # Find the abstract section
        abstract_match = re.search(
            r"AB\s+-\s+(.*?)(?=\n[A-Z]{2,4}\s+-\s+|\Z)", text, re.DOTALL
        )

        if not abstract_match:
            return None

        # Get the abstract text and clean it
        abstract_text = abstract_match.group(1).strip()

        # Join multiple lines and remove excess whitespace
        abstract_text = " ".join([line.strip() for line in abstract_text.split("\n")])

        return abstract_text

    def extract_mesh_terms(self, text) -> list:
        """
        Extract mesh terms from PubMed text.
        Mesh terms are tagged with "MH - " and can occur multiple times.
        """
        # Find all mesh terms
        mesh_matches = re.findall(r"MH\s+-\s+(.*?)(?=\n)", text)
        mesh_terms = [self.clean_mesh_term(term) for term in mesh_matches]

        # Return the list of mesh terms
        return mesh_terms

    def extract_pubmed_data(self, text):
        """
        Extract title, abstract, and mesh terms from PubMed text.
        Returns a dictionary with the extracted data.
        """
        return {
            "pmid": self.extract_pmid(text),
            "title": self.extract_title(text),
            "abstract": self.extract_abstract(text),
            "mesh_terms": self.extract_mesh_terms(text),
        }

    def extract_data_from_articles(self, articles: list[str]) -> dict:
        """
        Extracts data from a list of PubMed articles.
        """
        data = {}
        for article in tqdm(articles, desc="Extracting Pubmed data from articles"):
            extracted_data = self.extract_pubmed_data(article)
            self._distinct_mesh_terms.update(extracted_data["mesh_terms"])
            pmid = extracted_data["pmid"]
            if pmid:
                data[pmid] = extracted_data
        return data


class MeshTermFetcher:
    """
    Fetches MeSH term definitions from the MeSH database using the Entrez API.
    """

    def __init__(self) -> None:
        Entrez.email = ConfigEnv.ENTREZ_EMAIL
        self._db = "mesh"
        self._field = "MH"
        self._retmax = 1
        self._stop_terms = [
            "Year introduced:",
            "Subheadings:",
            "Tree Number(s)",
            "Previous Indexing:",
            "See Also:",
            "All MeSH Categories",
        ]
        self._file_name = "mesh_term_definitions.json"

    def get_mesh_ui(self, term: str) -> str:
        """
        Fetches the MeSH UI for a given MeSH term.
        """
        handle = Entrez.esearch(
            db=self._db,
            term=f'"{term}"[MeSH Terms]',
            field=self._field,
            retmax=self._retmax,
        )
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"][0] if record["IdList"] else None

    def extract_definition(self, term: str, text: str) -> str:
        text_lower = text.lower()
        term_lower = term.lower()
        start_index = text_lower.find(term_lower)
        if start_index == -1:
            start_index = 0
        # Determine the earliest occurrence of any stop term after the term
        stop_indexes = []
        for stop_term in self._stop_terms:
            idx = text.find(stop_term, start_index)
            if idx != -1:
                stop_indexes.append(idx)
        if stop_indexes:
            end_index = min(stop_indexes)
            definition = text[start_index:end_index].strip()
        else:
            definition = text[start_index:].strip()
        definition = definition.replace("[Subheading]", "")
        return definition

    def get_definition(self, ui: str, term: str):
        handle = Entrez.efetch(db=self._db, id=ui)
        text_data = handle.read()
        handle.close()
        return self.extract_definition(term, text_data)

    def fetch_definitions(self, mesh_terms: list):

        logger.debug(f"Working on file: {self._file_name}")
        mesh_definitions_file_path = os.path.join(
            ConfigPath.EXTERNAL_DATA_DIR, self._file_name
        )
        issued_definitions_file_path = os.path.join(
            ConfigPath.EXTERNAL_DATA_DIR, "issued_terms.json"
        )
        if os.path.exists(mesh_definitions_file_path):
            definitions = read_json_file(file_path=mesh_definitions_file_path)
            if not definitions:
                definitions = {}
        else:
            definitions = {}

        issued_terms = []
        for term in tqdm(mesh_terms):
            if term in definitions:
                continue
            try:
                ui = self.get_mesh_ui(term)
                if ui:
                    definition = self.get_definition(ui, term)
                    definitions[term] = definition
                else:
                    issued_terms.append(term)
            except Exception as e:
                logger.error(
                    f"Failed to retrieve MeSH definition for term: {term}. Error: {e}"
                )
                issued_terms.append(term)

            save_json_file(file_path=mesh_definitions_file_path, data=definitions)
            # save issued terms for retry
            save_json_file(file_path=issued_definitions_file_path, data=issued_terms)

        return definitions
