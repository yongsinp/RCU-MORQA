import logging
import os
from typing import Optional, Union, Tuple, List

import torch
from datasets import Dataset, tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, \
    DefaultDataCollator
from transformers.trainer_utils import get_last_checkpoint

from src.extraction.extractor import Extractor
from src.extraction.runner import ExtractionTask, run_tasks
from src.preprocess.data import Document, Label, QuestionType
from src.util.paths import DATA_RCU_EN_PATH, MODELS_PATH, OUT_PATH

MODEL_DIR = MODELS_PATH


class MRCExtractor(Extractor):
    IMPLICIT_QUESTION: dict[QuestionType, str] = {
        QuestionType.ADVICE: "What is your advice?",
        QuestionType.ASSESSMENT: "What is the assessment?",
        QuestionType.IDENTIFICATION: "Can you identify the problem?",
    }

    def __init__(self, model_name: str = "dmis-lab/biobert-v1.1", max_len: int = 256) -> None:
        """Initializes the MRC Extractor.

        Args:
            model_name: The name of the pre-trained model to use.
            max_len: The maximum sequence length for tokenization. Defaults to 256 as found in the training data (233).
        """
        super().__init__(model_name)
        self._tokenizer = None
        self.max_len = max_len
        self.device = self._get_device()
        self.fine_tuned_model = None

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Lazily loads the tokenizer."""
        if self._tokenizer is None:
            self.logger.info(f"Loading tokenizer for model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: AutoTokenizer) -> None:
        self._tokenizer = value

    def _get_device(self) -> str:
        """Determines the available device: cuda, mps, or cpu."""
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.logger.info(f"Using device: {device}")

        return device

    def _process_data(self, document: Document) -> dict:
        pairs: list[tuple[str, str, Optional[tuple[int, int]]]] = []

        # Create (question, response, answer span) pairs
        for q in document.questions:
            q_text = self.IMPLICIT_QUESTION[q.att.questtyp] if q.att.is_implicit else q.att.text

            for response in document.responses:
                answer_spans = [
                    (ann.start, ann.end)
                    for anns in response.annotations.values()
                    for ann in anns
                    if ann.label == Label.SHORTEST_ANSWER and q.att.id == ann.att.id
                ]

                if answer_spans:
                    for span in answer_spans:
                        pairs.append((q_text, response.content, span))
                else:
                    # Negative sample
                    pairs.append((q_text, response.content, None))

        if not pairs:
            return {}

        questions, responses, answer_spans = zip(*pairs)

        # Tokenize all pairs
        tokenized = self.tokenizer(
            list(questions),
            list(responses),
            max_length=self.max_len,
            truncation='only_second',  # Slide window over response
            stride=128,
            padding='max_length',
            return_overflowing_tokens=True,
            return_offsets_mapping=True  # Map tokens back to original text
        )

        # Map answer positions
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answer = answer_spans[sample_idx]

            # Negative samples point to the CLS token
            if answer is None:
                start_positions.append(0)
                end_positions.append(0)
                continue

            # Find the start and end of the context (sequence id: 1) in the tokenized input
            sequence_ids = tokenized.sequence_ids(i)
            try:
                context_start = sequence_ids.index(1)
                context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
            except ValueError:
                start_positions.append(0)
                end_positions.append(0)
                continue

            # If answer is not fully inside the context, label as CLS token
            char_start, char_end = answer
            if offsets[context_start][0] > char_start or offsets[context_end][1] < char_end:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Map Char Start -> Token Start
                token_start = context_start
                while token_start <= context_end and offsets[token_start][0] <= char_start:
                    token_start += 1
                start_positions.append(token_start - 1)

                # Map Char End -> Token End
                token_end = context_end
                while token_end >= context_start and offsets[token_end][1] >= char_end:
                    token_end -= 1
                end_positions.append(token_end + 1)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions

        return dict(tokenized)

    def _create_dataset(self, documents: list[Document]) -> Dataset:
        dataset = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "start_positions": [],
            "end_positions": []
        }

        for document in documents:
            if tokens := self._process_data(document):
                for k in dataset.keys():
                    # if k in tokens:
                    dataset[k].extend(tokens[k])

        return Dataset.from_dict(dataset)

    def train(
        self,
        train_documents,
        valid_documents,
        resume: bool = True,
        max_steps: int = -1,
    ) -> None:
        # Create datasets
        train_dataset = self._create_dataset(train_documents)
        valid_dataset = self._create_dataset(valid_documents)

        # Load model
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        trained_model_name = f"{self.model_name.replace('/', '_')}"

        args = TrainingArguments(
            output_dir=str(MODELS_PATH / trained_model_name),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            max_steps=max_steps,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=50,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=self.tokenizer,
            data_collator=DefaultDataCollator()
        )

        last_checkpoint = None
        if resume and os.path.isdir(args.output_dir):
            last_checkpoint = get_last_checkpoint(args.output_dir)

        trainer.train(resume_from_checkpoint=last_checkpoint)

        trainer.save_model(str(MODELS_PATH / f"{trained_model_name}_trained"))

    def _load_model_for_inference(self, model_path: str) -> None:
        self.logger.info(f"Loading model from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.fine_tuned_model = AutoModelForQuestionAnswering.from_pretrained(model_path)

        self.fine_tuned_model.to(self.device)
        self.fine_tuned_model.eval()

    def inference(self, question: str, response: str,
                  return_offsets: bool = False) -> Union[str, Tuple[str, int, int], None]:
        # Load fine-tuned model if not already loaded
        if self.fine_tuned_model is None:
            model_path = MODEL_DIR / f"{self.model_name.replace('/', '_')}_trained"
            self._load_model_for_inference(str(model_path))

        # Tokenize input
        inputs = self.tokenizer(
            question,
            response,
            return_tensors="pt",
            max_length=self.max_len,
            truncation='only_second',
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length',
        )

        offset_mapping = inputs.pop("offset_mapping")
        inputs.pop("overflow_to_sample_mapping")

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.fine_tuned_model(**inputs)

        best_score = float('-inf')
        best_answer = None

        # Iterate through every chunk
        for i in range(len(inputs['input_ids'])):
            start_logits = outputs.start_logits[i]
            end_logits = outputs.end_logits[i]

            # Find best span in this chunk
            start_idx = torch.argmax(start_logits).item()
            end_idx = torch.argmax(end_logits).item() + 1

            # Skip if no answer
            if start_idx == 0 or end_idx <= start_idx:
                continue

            score = start_logits[start_idx].item() + end_logits[end_idx - 1].item()

            if score > best_score:
                best_score = score

                # Decode answer text
                token_ids = inputs['input_ids'][i][start_idx:end_idx]
                answer_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

                if return_offsets:
                    # Calculate offsets
                    char_start = offset_mapping[i][start_idx][0].item()
                    char_end = offset_mapping[i][end_idx - 1][1].item()
                    best_answer = (answer_text, char_start, char_end)
                else:
                    best_answer = answer_text

        return best_answer

    def batch_inference(self, questions: list[str], responses: list[str], batch_size: int = 16,
                        return_offsets: bool = False) -> List[Union[str, Tuple[str, int, int], None]]:
        if self.fine_tuned_model is None:
            model_path = MODEL_DIR / f"{self.model_name.replace('/', '_')}_trained"
            self._load_model_for_inference(str(model_path))

        answers = []

        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_questions,
                batch_responses,
                return_tensors="pt",
                max_length=self.max_len,
                truncation='only_second',
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding='max_length',
            )

            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.fine_tuned_model(**inputs)

            batch_results = [None] * len(batch_questions)
            batch_scores = [float('-inf')] * len(batch_questions)

            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            for chunk_idx, sample_idx in enumerate(sample_map):
                s_logits = start_logits[chunk_idx]
                e_logits = end_logits[chunk_idx]

                # Find best span in this chunk
                start_token = torch.argmax(s_logits).item()
                end_token = torch.argmax(e_logits).item() + 1

                # Skip if no answer
                if start_token == 0 or end_token <= start_token:
                    continue

                score = s_logits[start_token].item() + e_logits[end_token - 1].item()

                if score > batch_scores[sample_idx]:
                    batch_scores[sample_idx] = score

                    # Decode text
                    token_ids = inputs['input_ids'][chunk_idx][start_token:end_token]
                    answer_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

                    # Calculate offsets
                    char_start = offset_mapping[chunk_idx][start_token][0].item()
                    char_end = offset_mapping[chunk_idx][end_token - 1][1].item()

                    batch_results[sample_idx] = (answer_text, char_start, char_end)

            # Add processed batch results to the final list
            for result in batch_results:
                if result is None:
                    answers.append(None)
                else:
                    if return_offsets:
                        answers.append(result)  # (text, start, end)
                    else:
                        answers.append(result[0])  # Just text

        return answers

    def extract_answers(self, document: Document) -> Document:
        """Extracts answers from the given document using the MRC model.

        Args:
            document: The Document to extract answers from.
        Returns:
            A new Document with extracted answer Annotations. All other attributes are copied from the input document except for shortest_answer annotations.
        """
        # Create a new document to return
        new_document = self._get_new_document(document, False, False, [Label.SHORTEST_ANSWER])

        # Create input
        qr_pairs = [(question, response) for question in new_document.questions for response in new_document.responses]
        questions = [self.IMPLICIT_QUESTION[q.att.questtyp] if q.att.is_implicit else q.att.text for q, _ in qr_pairs]
        responses = [r.content for _, r in qr_pairs]

        # Get answers
        answers = self.batch_inference(questions, responses, batch_size=16, return_offsets=True)

        # Create answer annotations
        for qr_pair, answer in zip(qr_pairs, answers):
            question, response = qr_pair
            question_id = question.att.id

            if answer is not None:
                answer_text, char_start, char_end = answer
                ann = self._create_annotation(
                    text=answer_text,
                    start=char_start,
                    end=char_end,
                    doc=f"{new_document.post_id}.ann",
                    label=Label.SHORTEST_ANSWER,
                    att_id=question_id
                )
                response.annotations['content'].append(ann)

        return new_document

    def _extract_questions(self, document: Document) -> Document:
        raise NotImplementedError("MRCExtractor only supports answer extraction.")


if __name__ == '__main__':
    path = str(DATA_RCU_EN_PATH)
    datasets = [
        "iiyi",
        "woundcare",
    ]
    splits = [
        "valid_gold",
        "valid_systems",
        "test_gold",
        "test_systems",
    ]

    extractor = MRCExtractor(model_name="dmis-lab/biobert-v1.1")

    run_tasks(
        extractor=extractor,
        data_path=path,
        out_path=str(OUT_PATH),
        datasets=datasets,
        splits=splits,
        tasks=(ExtractionTask("answer_extraction", "extract_answers", "Extracting Answers"),),
        model_name=extractor.model_name,
    )
