import argparse

from adaptor.evaluators.question_answering import ExtractiveQAEvaluator
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from common.dataset_selector import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="bert-base-cased", type=str,
                    help="A model path to evaluate")
args = parser.parse_args()

model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
dataset = get_dataset("QA", "ID", "validation", args.firstn)

evaluator = ExtractiveQAEvaluator()
evaluator(model, tokenizer, dataset)
