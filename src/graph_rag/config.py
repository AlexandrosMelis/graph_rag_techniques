import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


class Logger:
    @staticmethod
    def get_logger(
        name: str = "GraphRagLogger", log_file: str = None, level: int = logging.DEBUG
    ) -> logging.Logger:
        """
        Creates and returns a logger with the specified name and level.
        If log_file is provided, logs will also be written to the specified file.

        :param name: Name of the logger.
        :param log_file: Path to the log file.
        :param level: Logging level.
        :return: Configured logger.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(module)s - %(message)s"
        )
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        if not logger.handlers:
            logger.addHandler(ch)

            # If log_file is specified, add file handler
            if log_file:
                # Ensure the log directory exists
                log_path = Path(log_file).parent.parent
                log_path.mkdir(parents=True, exist_ok=True)

                fh = logging.FileHandler(log_file, encoding="utf-8")
                fh.setLevel(level)
                fh.setFormatter(formatter)
                logger.addHandler(fh)

        return logger


class ConfigEnv:
    """
    Central configuration class for environment variables.
    The variables are loaded at import time using `python-dotenv`
    (plus any already-existing OS environment variables).
    """

    # Load variables from .env file (if present) and system environment
    load_dotenv(find_dotenv(), override=True)

    ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
    ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY")
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DB = os.getenv("NEO4J_PUBMED_DATABASE")

    NEO4J_VARS = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DB")

    @classmethod
    def require(cls, *names: str) -> None:
        """
        Raise if any of the given variables is unset. Checked where a service is used,
        so the retrieval pipeline runs without Neo4j or NCBI credentials.
        """
        missing = [name for name in names if not getattr(cls, name)]
        if missing:
            message = f"Missing required environment variable(s): {', '.join(missing)}"
            logging.error(message)
            raise EnvironmentError(message)


class ConfigPath:
    """
    Central configuration class for all project directory paths.
    Responsible for creating each directory if it does not already exist.
    """

    PROJECT_DIR = Path(__file__).resolve().parents[2]
    BASE_DIR = Path(__file__).resolve().parent  # package dir

    # Data directories
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    INTERMEDIATE_DATA_DIR = os.path.join(DATA_DIR, "intermediate")
    EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external")
    RESULTS_DIR = os.path.join(DATA_DIR, "results")
    MODELS_DIR = os.path.join(DATA_DIR, "models")
    OUTPUT_DIR = os.path.join(DATA_DIR, "output")
    SPLITS_DIR = os.path.join(DATA_DIR, "splits")
    INDEX_DIR = os.path.join(DATA_DIR, "index")

    @classmethod
    def create_directories(cls):
        """Create each directory if it doesn't already exist."""
        dirs_to_create = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.EXTERNAL_DATA_DIR,
            cls.RESULTS_DIR,
            cls.INTERMEDIATE_DATA_DIR,
            cls.MODELS_DIR,
            cls.OUTPUT_DIR,
            cls.SPLITS_DIR,
            cls.INDEX_DIR,
        ]

        for directory in dirs_to_create:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Create all directories when this file is imported
ConfigPath.create_directories()
