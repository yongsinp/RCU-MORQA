import logging

from sentence_transformers import util

from src.extraction.extractor import Extractor
from src.extraction.runner import ExtractionTask, run_tasks
from src.extraction.sentence_boundary import SentenceSplitter
from src.preprocess.data import Document, Label
from src.similarity.biobert import BioBertSimilarity

logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


class BioBertExtractor(Extractor):
    """Extractor using BioBERT model."""

    def __init__(self, similarity_model_name: str, spacy_model_name: str = "en_core_web_sm",
                 ans_threshold: float = 0.5) -> None:
        """Initializes the BioBERT Extractor.

        Args:
            similarity_model_name: The name of the BioBERT model to use.
            spacy_model_name: The name of the spaCy model to use for sentence splitting.
        """
        super().__init__(similarity_model_name)
        self.spacy_model_name = spacy_model_name
        # Initialize models
        self.similarity_model = BioBertSimilarity(model_name=similarity_model_name)
        self.sentence_splitter = SentenceSplitter(model_name=spacy_model_name)

    def _extract_questions(self, document: Document) -> Document:
        raise NotImplementedError("Question extraction is not supported in BioBertExtractor.")

    def extract_answers(self, document: Document, threshold: float = 0.25) -> Document:
        """Extracts answers from the document using BioBERT.

        Args:
            document: The input Document to extract answers from.
            threshold: The similarity threshold for answer extraction.
        Returns:
            A new Document with extracted answer Annotations. All other attributes are copied from the input document except for shortest_answer annotations.
        """
        # Create a new document to return
        new_document = self._get_new_document(document, False, False, [Label.SHORTEST_ANSWER])

        responses = [self.sentence_splitter.split(response.content) for response in new_document.responses]
        response_vectors = [self.similarity_model.get_vectors(response) for response in responses]

        for qa_pair in new_document.qa_pairs.values():
            q_centroid = self.similarity_model.get_centroid_vector(
                [question.att.text for question in qa_pair.questions])

            for i, r_vectors in enumerate(response_vectors):
                if len(r_vectors) == 0:
                    continue

                similarities = util.cos_sim(q_centroid, r_vectors).flatten().tolist()
                idxs = [j for j, sim in enumerate(similarities) if sim >= threshold]

                for j in idxs:
                    text = responses[i][j]
                    start, end = self._find_spans(new_document.responses[i].content, text)
                    new_document.responses[i].annotations['content'].append(
                        self._create_annotation(text, start, end, f"{new_document.post_id}.ann", Label.SHORTEST_ANSWER)
                    )

        new_document.reset()

        return new_document

    def _get_answer_similarities(self, document: Document) -> tuple[list[float], list[float], list[float]]:
        """Calculates answer similarities for the given document."""
        responses = [self.sentence_splitter.split(response.content) for response in document.responses]
        ans_similarities, no_ans_similarities, not_ans_similarities = [], [], []

        for qa_pair in document.qa_pairs.values():
            q_centroid = self.similarity_model.get_centroid_vector(
                [question.att.text for question in qa_pair.questions])

            if qa_pair.answers:
                # Compute similarities for actual answers
                a_vectors = self.similarity_model.get_vectors([answer.att.text for answer in qa_pair.answers])
                ans_similarities.extend(util.cos_sim(q_centroid, a_vectors).flatten().tolist())

                # Compute similarities for non-answer responses
                not_ans_responses = []
                for i, response in enumerate(responses):
                    for sentence in response:
                        start, end = self._find_spans(document.responses[i].content, sentence)
                        for answer in qa_pair.answers:
                            if not (end <= answer.start or start >= answer.end):
                                break
                        else:
                            not_ans_responses.append(sentence)
                if not_ans_responses:
                    na_vectors = self.similarity_model.get_vectors(not_ans_responses)
                    not_ans_similarities.extend(util.cos_sim(q_centroid, na_vectors).flatten().tolist())
            # Compute similarities for no-answer responses
            else:
                for response in responses:
                    r_vectors = self.similarity_model.get_vectors(response)
                    no_ans_similarities.extend(util.cos_sim(q_centroid, r_vectors).flatten().tolist())

        return ans_similarities, no_ans_similarities, not_ans_similarities

    def get_answer_similarity_threshold(self, documents: list[Document]) -> tuple[dict, ...]:
        """Calculates answer similarity thresholds from the given documents.

        Args:
            documents: The list of Documents to calculate thresholds from.
        Returns:
            A tuple containing three dictionaries with 'avg', 'max', and 'min' thresholds for
            answer similarities, no-answer similarities, and not-answer similarities.
        """

        def calculate_thresholds(similarities: list[float]) -> dict:
            return {
                'avg': sum(similarities) / len(similarities),
                'max': max(similarities),
                'min': min(similarities),
                'cnt': len(similarities)
            }

        similarities = ([], [], [])

        for document in documents:
            for i, similarity in enumerate(self._get_answer_similarities(document)):
                similarities[i].extend(similarity)

        return tuple(calculate_thresholds(s) for s in similarities)


if __name__ == '__main__':
    extractor = BioBertExtractor(similarity_model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

    data_path = "../../data/rcu-en"
    out_path = "../../out"
    datasets = [
        "iiyi",
        "woundcare",
    ]
    splits = [
        "train_gold",
        "valid_gold",
    ]

    for threshold_int in range(25, 86, 5):
        threshold = threshold_int / 100
        run_tasks(
            extractor=extractor,
            data_path=data_path,
            out_path=out_path,
            datasets=datasets,
            splits=splits,
            tasks=(ExtractionTask(
                "answer_extraction",
                "extract_answers",
                f"Extracting Answers {threshold}",
                kwargs={"threshold": threshold},
            ),),
            model_name=f"BioBERT_{threshold}",
        )
