import copy
import dataclasses
import logging
import os

from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from src.extraction.llm import LlmExtractor
from src.preprocess.data import Document, QuestionType
from src.util.io import read_json, write_json

logging.getLogger('openai._base_client').setLevel(logging.WARNING)


class QwenExtractor(LlmExtractor):
    """Extractor using Alibaba Qwen models."""

    def __init__(self, model_name: str, max_output_tokens: int = 300) -> None:
        """Initializes the Qwen Extractor.

        Args:
            model_name: The name of the Qwen model to use.
            max_output_tokens: The maximum number of tokens to generate in the output.
        """
        super().__init__(model_name)
        self.max_output_tokens = max_output_tokens
        super().__init__(model_name, max_output_tokens)
        self.client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

    def _call_api(self, query: str, system_prompt: str) -> str:
        """Calls the Alibaba Qwen API."""
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
            modalities=["text"],
        )

        return response.choices[0].message.content

    def get_max_tokens(self, data: list[str]) -> int:
        """Gets the maximum number of tokens in the data using the Qwen tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained("../../external/qwen_tokenizer")
        return max(len(tokenizer.encode(text)) for text in data)
