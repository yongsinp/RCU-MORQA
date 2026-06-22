from functools import singledispatch
from typing import Any, Union

from src.eval.runner import run_iaa_classification_eval
from src.preprocess.data import Document, QuestionType
from src.util.paths import DATA_RCU_EN_PATH, OUT_PATH


@singledispatch
def eval(gold_data: Union[Document, list[Document]], pred_data: Union[Document, list[Document]]) -> dict[
    str, list[Any]]:
    """Evaluates the classification performance between gold and predicted data."""
    ...


@eval.register(Document)
def _(gold_data: Document, pred_data: Document) -> dict[str, list[Any]]:
    gold_problem = []
    pred_problem = []
    gold_test = []
    pred_test = []
    gold_treatment = []
    pred_treatment = []
    gold_follup = []
    pred_follup = []
    gold_severe = []
    pred_severe = []
    gold_conditional = []
    pred_conditional = []

    gold_responses = [ann
                      for response in gold_data.responses
                      for annotations in response.annotations.values()
                      for ann in annotations]
    pred_responses = [ann
                      for response in pred_data.responses
                      for annotations in response.annotations.values()
                      for ann in annotations]

    for gold, pred in zip(gold_responses, pred_responses):
        gold_problem.append(gold.att.is_prob)
        pred_problem.append(pred.att.is_prob)
        gold_test.append(gold.att.is_test)
        pred_test.append(pred.att.is_test)
        gold_treatment.append(gold.att.is_treat)
        pred_treatment.append(pred.att.is_treat)
        gold_follup.append(gold.att.is_follup)
        pred_follup.append(pred.att.is_follup)
        gold_severe.append(gold.att.is_severe)
        pred_severe.append(pred.att.is_severe)
        gold_conditional.append(gold.att.is_conditional)
        pred_conditional.append(pred.att.is_conditional)

    return {
        'gold_problem': gold_problem,
        'pred_problem': pred_problem,
        'gold_test': gold_test,
        'pred_test': pred_test,
        'gold_treatment': gold_treatment,
        'pred_treatment': pred_treatment,
        'gold_follup': gold_follup,
        'pred_follup': pred_follup,
        'gold_severe': gold_severe,
        'pred_severe': pred_severe,
        'gold_conditional': gold_conditional,
        'pred_conditional': pred_conditional,
    }


@eval.register(list)
def _(gold_data: list[Document], pred_data: list[Document]) -> dict[str, list[Any]]:
    gold_problem = []
    pred_problem = []
    gold_test = []
    pred_test = []
    gold_treatment = []
    pred_treatment = []
    gold_follup = []
    pred_follup = []
    gold_severe = []
    pred_severe = []
    gold_conditional = []
    pred_conditional = []

    for gold_doc, pred_doc in zip(gold_data, pred_data):
        result = eval(gold_doc, pred_doc)
        gold_problem.extend(result['gold_problem'])
        pred_problem.extend(result['pred_problem'])
        gold_test.extend(result['gold_test'])
        pred_test.extend(result['pred_test'])
        gold_treatment.extend(result['gold_treatment'])
        pred_treatment.extend(result['pred_treatment'])
        gold_follup.extend(result['gold_follup'])
        pred_follup.extend(result['pred_follup'])
        gold_severe.extend(result['gold_severe'])
        pred_severe.extend(result['pred_severe'])
        gold_conditional.extend(result['gold_conditional'])
        pred_conditional.extend(result['pred_conditional'])

    return {
        'gold_problem': gold_problem,
        'pred_problem': pred_problem,
        'gold_test': gold_test,
        'pred_test': pred_test,
        'gold_treatment': gold_treatment,
        'pred_treatment': pred_treatment,
        'gold_follup': gold_follup,
        'pred_follup': pred_follup,
        'gold_severe': gold_severe,
        'pred_severe': pred_severe,
        'gold_conditional': gold_conditional,
        'pred_conditional': pred_conditional,
    }


if __name__ == '__main__':
    # Paths
    gold_path = str(DATA_RCU_EN_PATH)
    pred_path = str(OUT_PATH / "medical_directives_classification")

    # File names
    datasets = [
        "iiyi",
        "woundcare",
    ]
    splits = [
        "test_gold",
        "test_systems",
    ]

    run_iaa_classification_eval(eval, gold_path, pred_path, datasets, splits)
