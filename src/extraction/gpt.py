import logging
from os import getenv

import tiktoken
from openai import AzureOpenAI

from src.extraction.llm import LlmExtractor
from src.extraction.runner import run_llm_tasks

logging.getLogger('openai._base_client').setLevel(logging.WARNING)


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
                    "content": f"Input: {query}\nOutput: "
                },
            ]
        )

        return response.choices[0].message.content

    def get_max_tokens(self, data: list[str]) -> int:
        """Gets the maximum token length in the data using tiktoken."""
        encoding = tiktoken.encoding_for_model(self.model_name)
        return max(len(encoding.encode(text)) for text in data)


if __name__ == '__main__':
    extractor = GptExtractor(model_name="gpt-4o", max_output_tokens=5000)

    data_path = "../../data/rcu-en"
    out_path = "../../out"
    datasets = [
        "iiyi",
        "woundcare",
    ]
    run_llm_tasks(
        extractor=extractor,
        data_path=data_path,
        out_path=out_path,
        datasets=datasets,
        question_splits=["train_gold", "valid_gold", "test_gold"],
        non_question_splits=["train_gold", "valid_gold", "valid_systems", "test_gold", "test_systems"],
    )
