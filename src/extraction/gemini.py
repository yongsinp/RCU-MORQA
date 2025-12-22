import copy
import dataclasses
import logging
import os

from google import genai
from google.genai import types
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

logging.getLogger('google_genai.models').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


class GeminiExtractor(LlmExtractor):
    """Extractor using Google Gemini models."""

    def __init__(self, model_name: str, max_output_tokens: int = 300, reasoning: bool = False) -> None:
        """Initializes the Gemini Extractor.

        Args:
            model_name: The name of the Gemini model to use.
            max_output_tokens: The maximum number of tokens to generate in the output.
            reasoning: Whether to enable reasoning capabilities.
        """
        super().__init__(model_name)
        self.max_output_tokens = max_output_tokens
        if "pro" in model_name.lower() and not reasoning:
            raise ValueError("Reasoning for Gemini Pro models can't be turned off")

        super().__init__(model_name, max_output_tokens + (8192 if reasoning else 0))
        self.client = genai.Client()
        self.config = types.GenerateContentConfig(
            max_output_tokens=self.max_output_tokens if self.max_output_tokens > 0 else None,
            # Reasoning
            thinking_config=types.ThinkingConfig(
                thinking_budget=None if reasoning else 0,
                include_thoughts=False,
            ),
            # Turn safety filters off
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
            ],
        )

    def _call_api(self, query: str, system_prompt: str) -> str:
        """Calls the Google Gemini API."""
        self.config.system_instruction = system_prompt
        response = self.client.models.generate_content(
            model=self.model_name,
            config=self.config,
            contents=f"Input: {query}\nOutput:"
        )
        return response.text

    def get_max_tokens(self, data: list[str]) -> int:
        """Gets the maximum token length in the data using Gemini tokenizer."""
        client = genai.Client()
        return max(client.models.count_tokens(model=self.model_name, contents=text).total_tokens for text in data)
