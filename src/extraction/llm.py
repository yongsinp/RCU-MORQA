import copy
import logging
import re
from abc import abstractmethod
from ast import literal_eval

from src.extraction.extractor import Extractor
from src.preprocess.data import Document, QuestionType, Polarity, Label

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
Output: {"polarity": "binary", "type": "assessment"}"""

SYSTEM_PROMPT_ANSWER = """You are a precise linguistic analysis engine specialized in medical context extraction. Your task is to process a list of "Responses" against a set of "Questions" (which share a single intent) and extract specific sentences that answer the questions.

INPUT DATA
You will receive:
1. Questions: A list of strings (phrased differently but asking the same thing).
2. Polarity: (binary, categorical, or open).
3. Question Type: (identification, assessment, advice, outcome_prediction).
4. Responses: A list of strings to analyze.

DEFINITIONS
Use these definitions strictly to determine relevance:
1. Polarity:
   - binary: Questions logically answerable with "Yes" or "No" (e.g., "Is it X?"). A relevant answer might not say "Yes/No" explicitly but provides the confirmation or refutation (e.g., "It is Y" implies "No" to "Is it X?").
   - categorical: Questions presenting specific choices.
   - open: Questions requiring description, explanation, or lists.

2. Question Type:
   - identification: Identifying the wound/disease cause, state, pathology, or name.
   - assessment: Evaluating severity, urgency, or current status.
   - advice: Actionable steps, tests, treatments, or prescriptions.
   - outcome_prediction: Predictions on recovery or permanent effects.

PROCESSING LOGIC
For each item in the "Responses" list, perform the following steps:

Step 1: Determine Relevance Strategy
   - IF the "Questions" list contains valid strings: You are looking for sentences that specifically address the semantic intent of those questions.
   - IF the "Questions" list is empty ("") or contains only empty strings: You are looking for any sentences within the response that match the provided "Question Type".

Step 2: Sentence-Level Extraction
Split the response into individual sentences. Analyze each sentence to see if it qualifies as an answer.
   - Condition A (Type Match): The sentence content must semantically align with the provided "Question Type". (e.g., If Question Type is "advice", but the sentence is a diagnosis/identification like "It is Eczema", it is NOT a match).
   - Condition B (Answer Match): If questions are provided, the sentence must directly answer the inquiry.
   - Note: For Binary Identification questions (e.g., "Is it Tinea?"), a sentence identifying a different disease (e.g., "It is Psoriasis") IS a valid answer because it implicitly answers "No".

Step 3: Formatting
   - Extract the qualifying sentences verbatim. Do not paraphrase.
   - Combine multiple relevant sentences from a single response with their original punctuation.
   - If no sentences in a response meet the criteria, the result for that index is an empty string "".
   - Maintain a strict one-to-one mapping with the input "Responses" list.

EXAMPLES
Example 1
Input: Question: ["Pygmy Moss?"], Polarity: binary, Type: identification, Response: ["Frictional Lichenoid Eruption", "Frictional lichenoid rash doesn't seem to be the case, but it appears to be some kind of lichenoid rash.", "Glossy Moss"] 
Output: ["Frictional Lichenoid Eruption", "Frictional lichenoid rash doesn't seem to be the case, but it appears to be some kind of lichenoid rash.", "Glossy Moss"]

Example 2
Input: Question: ["What skin disease?", "What is this emergency?"], Polarity: open, Type: identification, Response: ["Tension blister", "Papular urticaria", "For the itching diagnosis, a blister puncture will suffice.", "Papular urticaria, topical use of Lugen Shi lotion."]
Output: ["Tension blister", "Papular urticaria", "For the itching diagnosis, a blister puncture will suffice.", "Papular urticaria, topical use of Lugen Shi lotion."]

Example 3
Input: Q: [""], Polarity: open, Type: identification, Response: ["Elbow Black Acanthosis Nigricans", "Frictional Hyperkeratosis"]
Output: ["Elbow Black Acanthosis Nigricans", "Frictional Hyperkeratosis"]

Example 4
Input: Q: ["I suspect it's tinea, could an expert please confirm this?"], Polarity: binary, Type: identification, Response: ["The original poster is requested to provide more medical history. A sudden increase in blood count may suggest erysipelas, while a chronic process could indicate tuberculosis or even leprosy."]
Output: ["A sudden increase in blood count may suggest erysipelas, while a chronic process could indicate tuberculosis or even leprosy."]

Example 3
Input: Q: ["What should we do?"], Polarity: binary, Type: advice, Response: ["Early Stage of Eczema in Children", "Eczema", "Eczema...............", "Eczema is easy to treat for some, but not for others."]
Output: ["", "", "", ""]

OUTPUT FORMAT
Return strictly a JSON list of strings."""

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
