import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM

from configs.config import logger


class ChatModel:
    """
    **Groq models**
    Available models:
        - Production models:
            - llama-3.3-70b-versatile // 1,000 requests/day | 6,000 tokens/minute
            - llama3-70b-8192 // 1,000 requests/day | 6,000 tokens/minute
            - mixtral-8x7b-32768 // 14,400 requests/day | 5,000 tokens/minute
        - Preview models:
            - qwen-2.5-32b // 1,000 requests/day | 6,000 tokens/minute
            - deepseek-r1-distill-qwen-32b // 1,000 requests/day | 6,000 tokens/minute
            - deepseek-r1-distill-llama-70b // 1,000 requests/day | 6,000 tokens/minute
    ------------------------------------------
    **Google gemini models**
    Available models:
        - gemini-2.0-flash // 15 requests/minute
        - gemini-2.0-flash-lite // 30 requests/minute - 1,500 requests/day
    ------------------------------------------
    **Local Ollama Models**
    Available models:
        - gemma3:4b
        - deepseek-r1:7b
    """

    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name

        # available models
        self.groq_models = [
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
        ]
        self.google_models = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
        self.local_models = ["gemma3:4b", "deepseek-r1:7b"]

    def initialize_model(self) -> Any:
        if self.provider == "groq":
            if not os.environ.get("GROQ_API_KEY"):
                logger.debug("GROQ_API_KEY is not set")
                raise ValueError("GROQ_API_KEY is not set")
            if self.model_name not in self.groq_models:
                logger.debug(f"Model {self.model_name} not supported")
                raise ValueError(f"Model {self.model_name} not supported")
            llm = init_chat_model(
                self.model_name, model_provider=self.provider, temperature=0
            )
        elif self.provider == "google":
            if not os.environ.get("GOOGLE_API_KEY"):
                logger.debug("GOOGLE_API_KEY is not set")
                raise ValueError("GOOGLE_API_KEY is not set")
            if self.model_name not in self.google_models:
                logger.debug(f"Model {self.model_name} not supported")
                raise ValueError(f"Model {self.model_name} not supported")
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=3,
            )
        elif self.provider == "ollama":
            if self.model_name not in self.local_models:
                logger.debug(f"Model {self.model_name} not supported")
                raise ValueError(f"Model {self.model_name} not supported")
            llm = OllamaLLM(model=self.model_name)
        else:
            raise ValueError(f"Provider {self.provider} not supported")
        logger.debug(f"Initialized model {self.model_name}")
        return llm
