import os

from configs import ConfigPath
from utils.utils import save_json_file


class DatasetConstructor:

    def __init__(self, bioasq_data: list[dict], pubmed_data: dict):
        self.bioasq_data = bioasq_data
        self.pubmed_data = pubmed_data

    def create_graph_data(self):
        """
        Combines the bioasq and pubmed data to create a json, which can be used to stored
        in graph representation in Neo4j Database.
        """
        bioasq_graph_data = self.bioasq_data.copy()

        for bioasq_sample in bioasq_graph_data:

            relevant_pmids = bioasq_sample.pop("relevant_passage_ids")
            # get relevant pubmed data
            contexts = [
                self.pubmed_data[str(pmid)]
                for pmid in relevant_pmids
                if pmid in self.pubmed_data
            ]

            bioasq_sample["articles"] = contexts

        # save the graph data
        save_json_file(
            file_path=os.path.join(ConfigPath.RAW_DATA_DIR, "bioasq_graph_data.json"),
            data=bioasq_graph_data,
        )

        return bioasq_graph_data
