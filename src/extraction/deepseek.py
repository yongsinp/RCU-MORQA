import logging
import os
from time import sleep

from openai import OpenAI
from transformers import AutoTokenizer

from src.extraction.llm import LlmExtractor
from src.util.paths import DEEPSEEK_TOKENIZER_PATH

logging.getLogger('openai._base_client').setLevel(logging.WARNING)


class DeepSeekExtractor(LlmExtractor):
    """Extractor using DeepSeek models."""

    def __init__(self, model_name: str, max_output_tokens: int = 300) -> None:
        """Initializes the DeepSeek Extractor.

        Args:
            model_name: The name of the DeepSeek model to use.
            max_output_tokens: The maximum number of tokens to generate in the output.
        """
        super().__init__(model_name)
        self.max_output_tokens = max_output_tokens
        super().__init__(model_name, max_output_tokens)
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com",
        )

    def _call_api(self, query: str, system_prompt: str) -> str:
        """Calls the DeepSeek API."""
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
                    "content": f"Input: {query}\nOutput: "
                },
            ],
            stream=False
        )

        return response.choices[0].message.content

    def get_max_tokens(self, data: list[str]) -> int:
        """Gets the maximum number of tokens in the data using the DeepSeek tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(str(DEEPSEEK_TOKENIZER_PATH))
        return max(len(tokenizer.encode(text)) for text in data)


class AzureDeepSeekExtractor(LlmExtractor):
    """Extractor using Azure DeepSeek models."""

    def __init__(self, model_name: str, max_output_tokens: int = 300) -> None:
        """Initializes the DeepSeek Extractor.

        Args:
            model_name: The name of the DeepSeek model to use.
            max_output_tokens: The maximum number of tokens to generate in the output.
        """
        super().__init__(model_name)
        self.max_output_tokens = max_output_tokens
        super().__init__(model_name, max_output_tokens)
        self.client = OpenAI(
            base_url="https://uwclms-resource.services.ai.azure.com/openai/v1/",
            api_key=os.getenv("AZURE_API_KEY"),
        )

    def _call_api(self, query: str, system_prompt: str) -> str:
        """Calls the Azure DeepSeek API."""
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
            ],
            stream=False
        )

        sleep(3)  # To avoid rate limiting

        return response.choices[0].message.content

    def get_max_tokens(self, data: list[str]) -> int:
        """Gets the maximum number of tokens in the data using the DeepSeek tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(str(DEEPSEEK_TOKENIZER_PATH))
        return max(len(tokenizer.encode(text)) for text in data)
