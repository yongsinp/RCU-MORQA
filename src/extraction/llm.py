import copy
import logging
import re
from abc import abstractmethod
from ast import literal_eval

from src.extraction.extractor import Extractor
from src.preprocess.data import Document, QuestionType, Polarity, Label
from src.prompts import SYSTEM_PROMPT_QUESTION, SYSTEM_PROMPT_CLASSIFICATION, SYSTEM_PROMPT_ANSWER, SYSTEM_PROMPT_IAA

logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


class LlmExtractor(Extractor):
    """Abstract base class for LLM-based question extractors and classifiers."""

    def __init__(self, model_name: str, max_output_tokens: int = -1) -> None:
        """Initializes the LLM extractor.

        Args:
            model_name: The name of the LLM model to use.
            max_output_tokens: The maximum number of tokens to generate in the output.
        """
        super().__init__(model_name)
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _find_spans(text: str, subtext: str) -> tuple[int, int]:
        """Finds the start and end indices of subtext in text, case-insensitive.

        Args:
            text: The main text to search within.
            subtext: The subtext to find.
        Returns:
            A tuple containing the start and end indices of the subtext in the text. If not found, returns (0, 0).
        """
        text = text.lower()
        subtext = subtext.lower()

        for i in range(len(text) - len(subtext) + 1):
            if text[i:i + len(subtext)] == subtext:
                return i, i + len(subtext)

        return 0, 0

    @staticmethod
    def _get_outermost_list(text: str) -> str:
        """Extracts the outermost list from a string representation of a list.

        Args:
            text: The string to extract the list from.
        Returns:
            The extracted list as a string.
        Raises:
            ValueError: If failed to extract the list.
        """

        start = text.find('[')
        depth = 0

        for i in range(start, len(text)):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        raise ValueError("Failed to extract outermost list from text")

    @abstractmethod
    def _call_api(self, text: str, system_prompt: str) -> str:
        """Calls the LLM API with the given text and system prompt.

        Args:
            text: The input text to process.
            system_prompt: The system prompt to guide the LLM.
        Returns:
            The LLM's response as a string.
        """
        ...

    @abstractmethod
    def get_max_tokens(self, data: list[str]) -> int:
        """Calculates the maximum number of tokens in the given data.

        Args:
            data: A list of strings to analyze.
        Returns:
            The maximum number of tokens found in the data.
        """
        ...

    def _extract_questions(self, document: Document) -> Document:
        """Extracts questions from the given document using the LLM.

        Args:
            document: The Document to extract questions from.
        Returns:
            A new Document with extracted question Annotations. All other attributes are copied from the input document except for annotations and responses.
        """
        # Create a new document to return
        new_document = self._get_new_document(document)

        # Extract questions using LLM
        for key in ('query_title', 'query_content'):
            text = getattr(document, key).strip()
            if not text:
                continue

            try:
                response = self._call_api(text, SYSTEM_PROMPT_QUESTION)
            except Exception as e:
                self.logger.error("API error ({}): {}".format(document.post_id, e))
                continue

            annotations = []
            try:
                match = re.search(r'\[.*?\]', response, re.S)
                json_str = match.group(0) if match else '[]'
                json_str = re.sub(r"(\w)'(s|re|ve|ll|d|m|t)\b", r"\1\'\2", json_str, flags=re.IGNORECASE)
                extractions = literal_eval(json_str)

                for extraction in extractions:
                    start, end = self._find_spans(text, extraction)
                    if start == end:
                        extraction = extraction[:-1]  # Try without trailing punctuation
                        start, end = self._find_spans(text, extraction)
                    if start != end:
                        annotations.append(self._create_annotation(extraction, start, end))
                    else:
                        self.logger.error("Span not found ({}): {}".format(document.post_id, extraction))
            except Exception as e:
                self.logger.error("JSON parsing error: {}\n{}".format(e, response))

            new_document.annotations[key] = annotations

        return new_document

    def _classify_question(self, question: str) -> dict:
        """Classifies a question's polarity and type using the LLM.

        Args:
            question: The text of the question to classify.
        Returns:
            A dictionary with 'polarity' and 'type' if classification is successful, otherwise an empty dict.
        """
        try:
            response = self._call_api(question, SYSTEM_PROMPT_CLASSIFICATION)
        except Exception as e:
            self.logger.error("API error: {}".format(e))
            return {}

        try:
            match = re.search(r'\{.*?\}', response, re.S)
            json_str = match.group(0) if match else '{}'
            json_str = json_str.replace("'", '"')
            classes = literal_eval(json_str)

            # Clean up keys and values
            cleaned_classes = {}
            for k, v in classes.items():
                k_clean = k.strip().lower()
                v_clean = v.strip().lower()

                if k_clean == 'polarity':
                    cleaned_classes[k_clean] = v_clean if v_clean in Polarity else None
                elif k_clean == 'type':
                    cleaned_classes[k_clean] = v_clean if v_clean in QuestionType else None

            return cleaned_classes
        except Exception as e:
            self.logger.error("JSON parsing error: {}\n{}".format(e, response))
            return {}

    def classify_questions(self, document: Document) -> Document:
        """Classifies the questions in the given document using the LLM.

        Args:
            document: The Document containing questions to classify.
        Returns:
            A new Document with extracted question Annotations. All other attributes are copied from the input document except for annotations.
        """
        new_doument = self._get_new_document(document, clear_annotations=False)

        for question in new_doument.questions:
            # Implicit questions have 'open' polarity and can have any of the question types defined in `Attribute.IMPLICIT_QUESTTYP`
            if question.att.is_implicit:
                continue

            # We don't have enough instances of NOT_CC questions
            if question.att.questtyp == QuestionType.NOT_CC:
                continue

            pred = self._classify_question(question.att.text)
            question.att.polarity = pred.get('polarity')
            question.att.questtyp = pred.get('type')

        return new_doument

    def extract_answers(self, document: Document) -> Document:
        """Extracts answers from the given document using the LLM.

        Args:
            document: The Document to extract answers from.
        Returns:
            A new Document with extracted answer Annotations. All other attributes are copied from the input document except for response annotations.
        """
        new_document = copy.deepcopy(document)

        responses = []
        for response in new_document.responses:
            # Collect response texts
            responses.append(response.content)
            # Clear response annotations
            response.annotations = {key: [] for key in response.annotations}

        for id_, qa_pair in new_document.qa_pairs.items():
            # Create input
            questions = [q.att.text for q in qa_pair.questions]
            polarity = str(qa_pair.questions[0].att.polarity)
            questtyp = str(qa_pair.questions[0].att.questtyp)
            input = f"Question: {questions}, Polarity: {polarity}, Type: {questtyp}, Response: {responses}"

            try:
                llm_response = self._call_api(input, SYSTEM_PROMPT_ANSWER)
            except Exception as e:
                self.logger.error("API error: {}".format(e))
                continue

            answers = []
            try:
                match = re.search(r'\[.*?\]', llm_response, re.S)
                json_str = match.group(0) if match else '[]'
                json_str = re.sub(r"(\w)'(s|re|ve|ll|d|m|t)\b", r"\1\'\2", json_str, flags=re.IGNORECASE)
                extractions = literal_eval(json_str)

                for response, extraction in zip(responses, extractions):
                    start, end = self._find_spans(response, extraction)
                    if start != end:
                        answers.append(self._create_annotation(extraction, start, end, doc=f"{document.post_id}.ann",
                                                               label=Label.SHORTEST_ANSWER, att_id=id_))
                    else:
                        answers.append(None)
                        if end > 0:
                            self.logger.error("Span not found ({}): {}".format(document.post_id, extraction))
            except Exception as e:
                self.logger.error("JSON parsing error: {}\n{}".format(e, llm_response))

            for response, answer in zip(new_document.responses, answers):
                if answer:
                    response.annotations['content'].append(answer)

        return new_document
