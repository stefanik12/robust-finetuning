import argparse
from typing import Any

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForSequenceClassification

from common.dataset_selector import get_dataset
from evaluation.evaluators import Evaluator
from evaluation.predictors import FinetunedQAHeadPredictor, FinetunedClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="bert-base-cased", type=str,
                    help="A model path to evaluate")
parser.add_argument("--task", default="QA", type=str,
                    help="Task to evaluate on. One of 'QA', 'NLI', 'Paraphrases'.")
parser.add_argument("--metric", default="exact_match", type=str,
                    help="A metric to report.")
parser.add_argument("--append_results_to_file", default=False, type=str,
                    help="A text file where to append the evaluation results")
parser.add_argument("--firstn", default=False, type=int,
                    help="A number of first-n samples from the set to evaluate on.")
args = parser.parse_args()


def report(results_type: str, results: Any) -> None:
    report_str = str({"task": args.task, "type": results_type, "model": args.model_path,
                      "metric": args.metric, "results": results})
    print(report_str)

    if args.append_results_to_file:
        with open(args.append_results_to_file, mode="a") as f:
            print(report_str, file=f)


if args.task == "QA":
    # TODO: QA preprocessing is problematic

    id_dataset = get_dataset("QA", "ID", "validation", args.firstn)
    ood_dataset = get_dataset("QA", "OOD", "validation", args.firstn)

    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    evaluator = Evaluator()
    predictor = FinetunedQAHeadPredictor(model, tokenizer)

    id_results = evaluator.evaluate(args.metric, predictor, id_dataset)
    report("ID", id_results)
    ood_results = evaluator.evaluate(args.metric, predictor, ood_dataset)
    report("OOD", ood_results)

elif args.task in ("NLI", "Paraphrasing"):
    id_dataset = get_dataset(args.task, "ID", "validation", args.firstn)
    ood_dataset = get_dataset(args.task, "OOD", "validation", args.firstn)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    evaluator = Evaluator()
    predictor = FinetunedClassifier(model, tokenizer)

    id_results = evaluator.evaluate(args.metric, predictor, id_dataset)
    report("ID", id_results)
    ood_results = evaluator.evaluate(args.metric, predictor, ood_dataset)
    report("OOD", ood_results)

else:
    raise ValueError("Unknown task %s" % args.task)

