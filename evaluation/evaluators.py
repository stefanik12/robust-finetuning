import abc
from typing import List

from datasets import Dataset

from predictors import Predictor


class Evaluator:
    
    def exact_match(self, predicted: List[str], actual: List[str]) -> float:
        assert len(predicted) == len(actual), "A number of predictions and references must match."
        matching = sum(p == a for p, a in zip(predicted, actual))
        total = len(predicted)
        return matching / total

    def evaluate(self, metric: str, predictor: Predictor, dataset: Dataset) -> float:
        predictions, references = predictor.predict(dataset)
        if metric == "exact_match":
            return self.exact_match(predictions, references)
        else:
            raise ValueError(metric)
