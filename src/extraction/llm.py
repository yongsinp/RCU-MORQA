import logging
import re
from abc import abstractmethod
from ast import literal_eval

from src.extraction.extractor import Extractor
from src.preprocess.data import Document, QuestionType

SYSTEM_PROMPT_QUESTION = """You are an advanced Medical Scribe.

    Your task is to identify and extract all text segments where the user is soliciting medical advice, diagnosis, opinion, or help.

    GUIDELINES
    1. Focus on Intent: Extract any text where the user is seeking an answer or a solution. This includes direct questions (e.g., Is this normal?) and implicit requests (e.g., Please help me identify this).
    2. Verbatim Extraction: Extract the text exactly as it appears in the source, including the trailing punctuations, if any.
    3. Context: Split compound sentences. If a user asks "What is this and how do I treat it?", extract them as multiple entries.

    OUTPUT FORMAT
    Return a JSON object just containing a list of strings. If no inquiries are found, return an empty list.

    EXAMPLES
    Input: Urgent!!! Is this Dermatitis due to Blattella???
    Output: ["Is this Dermatitis due to Blattella???"]

    Input: The patient is a 49-year-old female with papules on her face. She has a history of rosacea.
    Output: []

    Input: Lower limb eczema (with picture), please provide diagnosis and prescription.
    Output: ["please provide diagnosis", "prescription"]"""

SYSTEM_PROMPT_CLASSIFICATION = """You are an expert Medical Linguistic Analyzer.

Your task is to classify a given medical question based on two specific dimensions: Polarity and Type.

DEFINITIONS
1. Polarity:
   - binary: Questions that can be logically answered with a simple "Yes" or "No" (e.g., "hyperkeratosis, can it appear on the lower limbs?").
   - categorical: Questions presenting a choice between specific options (e.g., "Oral mucosal disease, eczema? Herpes?", "Is this urticaria or a skin allergy?").
   - open: Questions requiring a descriptive response, explanation, or list (Who, What, Where, When, Why, How) (e.g., "what tests need to be done.").

2. Type:
   - identification: Asking to identify the wound’s cause, current state, pathology, or developments (e.g., "please provide diagnosis", "May I ask what kind of skin disease is this?").
   - assessment: Asking to evaluate severity or urgency (e.g., "Is this a serious issue?", "Is there a problem with this wound?").
   - advice: Asking for actionable medical steps, including tests, treatments, or prescriptions (e.g., "Treatment for Chronic Urticaria", "Can someone give me a suggestion?").
   - outcome_prediction: Asking for predictions regarding recovery time or permanent effects (e.g., "How many days will it take to cure this disease approximately?", "Will this lead to tetanus?").

OUTPUT FORMAT
Return a single JSON object containing "polarity" and "type".

EXAMPLES
Input: Please help to identify what this is on my hand.
Output: {"polarity": "open", "type": "identification"}

Input: What kind of topical medication works best?
Output: {"polarity": "open", "type": "advice"}

Input: Is it eczema or acute impetigo-like pityriasis versicolor?
Output: {"polarity": "categorical", "type": "identification"}

Input: how long will it take to heal?
Output: {"polarity": "open", "type": "outcome_prediction"}

Input: Will it heal without deforming?
Output: {"polarity": "binary", "type": "outcome_prediction"}

Input: is the condition severe?
Output: {"polarity": "binary", "type": "assessment"}
"""

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
            return classes if 'polarity' in classes and 'type' in classes else {}
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
