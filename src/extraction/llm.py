import logging
import re
from abc import abstractmethod
from ast import literal_eval
from copy import deepcopy

from src.extraction.extractor import Extractor
from src.preprocess.data import Document, QuestionType, Polarity, Label, Attribute
from src.prompts import SYSTEM_PROMPT_QUESTION_EXTRACTION, SYSTEM_PROMPT_QUESTION_CLASSIFICATION, \
    SYSTEM_PROMPT_ANSWER_EXTRACTION, SYSTEM_PROMPT_IAA_EXTRACTION, \
    SYSTEM_PROMPT_PROGNOSIS_EXTRACTION, SYSTEM_PROMPT_IAA_CLASSIFICATION

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

    def _get_llm_response(self, llm_input: str, system_prompt: str) -> str:
        """Gets the LLM response for the given input and system prompt."""
        try:
            return self._call_api(llm_input, system_prompt)
        except Exception as error:
            self.logger.error(f"API error: {error}")
            return ""

    def _extract_list_from_llm_response(self, llm_response: str) -> list:
        """Parses the LLM response to extract a list."""
        try:
            json_str = self._get_outermost_list(llm_response)
            json_str = re.sub(r"(\w)'(s|re|ve|ll|d|m|t)\b", r"\1\'\2", json_str, flags=re.IGNORECASE)
            return literal_eval(json_str)
        except Exception as e:
            self.logger.error("JSON parsing error: {}\n{}".format(e, llm_response))
            return []

    def _extract_questions(self, document: Document) -> Document:
        """Extracts questions from the given document using the LLM.

        Args:
            document: The Document to extract questions from.
        Returns:
            A new Document with extracted question Annotations. All other attributes are copied from the input document except for Annotations and Responses.
        """
        # Create a new document to return
        # Responses are cleared because they are longer relevant after new questions are extracted
        new_document = self._get_new_document(document)

        for key in ('query_title', 'query_content'):
            text = getattr(document, key).strip()
            if not text:
                continue

            # Extract questions using LLM
            llm_response: str = self._get_llm_response(text, SYSTEM_PROMPT_QUESTION_EXTRACTION)
            extractions: list[str] = self._extract_list_from_llm_response(llm_response)

            # Create question annotations
            annotations = []

            for extraction in extractions:
                start, end = self._find_spans(text, extraction)
                if start == end:
                    extraction = extraction[:-1]  # Retry without trailing punctuation
                    start, end = self._find_spans(text, extraction)
                if start != end:
                    annotations.append(self._create_annotation(extraction, start, end))
                else:
                    self.logger.error("Span not found ({}): {}".format(document.post_id, extraction))

            new_document.annotations[key] = annotations

        return new_document

    def _classify_question(self, question: str) -> dict:
        """Classifies a question's polarity and type using the LLM.

        Args:
            question: The text of the question to classify.
        Returns:
            A dictionary with 'polarity' and 'type' if classification is successful, otherwise an empty dict.
        """
        # Get LLM response
        llm_response: str = self._get_llm_response(question, SYSTEM_PROMPT_QUESTION_CLASSIFICATION)
        if not llm_response:
            return {}

        try:
            # Extract dictionary from LLM response
            match = re.search(r'\{.*?\}', llm_response, re.S)
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
        except Exception as error:
            self.logger.error("JSON parsing error: {}\n{}".format(error, llm_response))
            return {}

    def classify_questions(self, document: Document) -> Document:
        """Classifies the questions in the given document using the LLM.

        Args:
            document: The Document containing questions to classify.
        Returns:
            A new Document with extracted question Annotations. All other attributes are copied from the input document except for annotations.
        """
        # Create a new document to return
        new_document = self._get_new_document(document, clear_annotations=False, clear_responses=False)

        for question in new_document.questions:
            # Implicit questions have 'open' polarity and can have any of the question types defined in `Attribute.IMPLICIT_QUESTTYP`
            if question.att.is_implicit:
                continue

            # We don't have enough instances of NOT_CC questions
            if question.att.questtyp == QuestionType.NOT_CC:
                continue

            # Classify question using LLM
            pred = self._classify_question(question.att.text)
            question.att.polarity = pred.get('polarity')
            question.att.questtyp = pred.get('type')

        return new_document

    def extract_answers(self, document: Document) -> Document:
        """Extracts answers from the given document using the LLM.

        Args:
            document: The Document to extract answers from.
        Returns:
            A new Document with extracted answer Annotations. All other attributes are copied from the input document except for shortest_answer annotations.
        """
        # Create a new document to return
        new_document = self._get_new_document(document, False, False, [Label.SHORTEST_ANSWER])
        responses = [response.content for response in new_document.responses]

        for id_, qa_pair in new_document.qa_pairs.items():
            # Create input
            questions = [q.att.text for q in qa_pair.questions]
            polarity = str(qa_pair.questions[0].att.polarity)
            questtyp = str(qa_pair.questions[0].att.questtyp)
            input_ = f"Questions: {questions}, Polarity: {polarity}, Type: {questtyp}, Response: {responses}"

            # Extract answers using LLM
            llm_response: str = self._get_llm_response(input_, SYSTEM_PROMPT_ANSWER_EXTRACTION)
            extractions: list[str] = self._extract_list_from_llm_response(llm_response)

            # Create annotations
            answers = []
            for response, extraction in zip(responses, extractions):
                start, end = self._find_spans(response, extraction)
                if start != end:
                    answers.append(self._create_annotation(extraction, start, end, doc=f"{document.post_id}.ann",
                                                           label=Label.SHORTEST_ANSWER, att_id=id_))
                else:
                    answers.append(None)
                    if end > 0:
                        self.logger.error("Span not found ({}): {}".format(document.post_id, extraction))

            # Add annotations to responses
            for response, answer in zip(new_document.responses, answers):
                if answer:
                    response.annotations['content'].append(answer)

        return new_document

    def extract_iaa(self, document: Document) -> Document:
        """Extracts Medical IAA annotations from the given document using the LLM.

        Args:
            document: The Document to extract Medical IAA annotations from.
        Returns:
            A new Document with extracted Medical IAA Annotations. All other attributes are copied from the input document except for medical_iaa annotations.
        """
        # Create a new document to return
        new_document = self._get_new_document(document, False, False, [Label.MEDICAL_IAA])

        # Extract Medical IAAs using LLM
        responses = [response.content for response in new_document.responses]
        llm_response: str = self._get_llm_response(str(responses), SYSTEM_PROMPT_IAA_EXTRACTION)
        extractions: list[str] = self._extract_list_from_llm_response(llm_response)

        # Create annotations
        annotations = []
        for response, extraction in zip(responses, extractions):
            if not extraction:
                annotations.append([])
                continue

            iaas = []

            for item in extraction:
                start, end = self._find_spans(response, item)
                if start != end:
                    iaas.append(self._create_annotation(item, start, end, doc=f"{document.post_id}.ann",
                                                        label=Label.MEDICAL_IAA))
                else:
                    if end > 0:
                        self.logger.error("Span not found ({}): {}".format(document.post_id, item))

            annotations.append(iaas)

        # Add annotations to responses
        for response, ann in zip(new_document.responses, annotations):
            response.annotations['content'].extend(ann)

        return new_document

    def classify_iaa(self, document: Document) -> Document:
        """Classifies the Medical IAA annotations in the given document using the LLM.

        Args:
            document: The Document containing Medical IAA annotations to classify.
        Returns:
            A new Document with classified Medical IAA Annotations. All other attributes are copied from the input document except for medical_iaa annotations.
        """

        def set_attribute(att: Attribute, pred: dict) -> None:
            labels = pred.get('labels', [])
            if not labels:
                self.logger.error(f"No labels predicted for IAA: Document ID {document.post_id}, Ent ID: {iaa.ent_id}")

            # Set attributes based on labels and predictions
            att.is_follup = 'followup' in labels
            att.is_prob = 'problem' in labels
            att.is_test = 'test' in labels
            att.is_treat = 'treatment' in labels
            att.is_conditional = bool(pred.get('is_conditional', False))
            att.is_severe = bool(pred.get('is_severe', False))

            # 'is_conditional' can only be True if 'followup' label is present
            if att.is_conditional and not att.is_follup:
                self.logger.error(
                    f"Conflict in IAA attributes: 'is_conditional' is True but 'followup' label is missing. "
                    f"Document ID {document.post_id}, Ent ID: {iaa.ent_id}")

        # Create a new document to return
        new_document = deepcopy(document)

        for response in new_document.responses:
            iaas = [ann for anns in response.annotations.values() for ann in anns if ann.label == Label.MEDICAL_IAA]
            # Reset attributes to only contain text
            for iaa in iaas:
                iaa.att = Attribute(text=iaa.att.text)

            # Skip if no IAAs are present
            if not iaas:
                continue

            # Create input
            input_ = f'Context: "{response.content}", IAA Texts: {[iaa.att.text for iaa in iaas]}'

            # Classify IAAs using LLM
            llm_response: str = self._get_llm_response(input_, SYSTEM_PROMPT_IAA_CLASSIFICATION)
            preds = self._extract_list_from_llm_response(llm_response)

            # Assign attributes to IAAs
            for iaa, pred in zip(iaas, preds):
                set_attribute(iaa.att, pred)

        return new_document

    def extract_prognosis(self, document: Document) -> Document:
        """Extracts prognosis annotations from the given document using the LLM.

        Args:
            document: The Document to extract prognosis annotations from.
        Returns:
            A new Document with extracted prognosis Annotations. All other attributes are copied from the input document except for prognosis annotations.
        """
        # Create a new document to return
        new_document = self._get_new_document(document, False, False, [Label.PROGNOSIS])

        # Extract prognoses using LLM
        responses = [response.content for response in new_document.responses]
        llm_response: str = self._get_llm_response(str(responses), SYSTEM_PROMPT_PROGNOSIS_EXTRACTION)
        extractions: list[str] = self._extract_list_from_llm_response(llm_response)

        # Create annotations
        annotations = []
        for response, extraction in zip(responses, extractions):
            if not extraction:
                annotations.append([])
                continue

            prognoses = []

            for item in extraction:
                start, end = self._find_spans(response, item)
                if start != end:
                    prognoses.append(self._create_annotation(item, start, end, doc=f"{document.post_id}.ann",
                                                             label=Label.PROGNOSIS))
                else:
                    if end > 0:
                        self.logger.error("Span not found ({}): {}".format(document.post_id, item))

            annotations.append(prognoses)

        # Add annotations to responses
        for response, ann in zip(new_document.responses, annotations):
            response.annotations['content'].extend(ann)

        return new_document
