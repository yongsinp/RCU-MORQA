import copy
import dataclasses
import logging
import os
from os import getenv

import tiktoken
from openai import AzureOpenAI
from tqdm import tqdm

from src.extraction.llm import LlmExtractor
from src.preprocess.data import Document, QuestionType
from src.util.io import read_json, write_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logging.getLogger('openai._base_client').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


class GptExtractor(LlmExtractor):
    """Extractor using Azure OpenAI GPT models."""

    def __init__(self, model_name: str, max_output_tokens: int = 300) -> None:
        """Initializes the GPT Extractor.

        Args:
            model_name: The name of GPT model to use.
            max_output_tokens: The maximum number of tokens to generate in the output.
        """
        super().__init__(model_name)
        self.max_output_tokens = max_output_tokens
        super().__init__(model_name, max_output_tokens)
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint="https://uwclms-resource.cognitiveservices.azure.com/",
            api_key=getenv("AZURE_API_KEY"),
        )

    def _call_api(self, query: str, system_prompt: str) -> str:
        """Calls the Azure OpenAI GPT API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=self.max_output_tokens,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Input: {query}\nOutput:"
                },
            ]
        )

        return response.choices[0].message.content

    def get_max_tokens(self, data: list[str]) -> int:
        """Gets the maximum token length in the data using tiktoken."""
        encoding = tiktoken.encoding_for_model(self.model_name)
        return max(len(encoding.encode(text)) for text in data)
