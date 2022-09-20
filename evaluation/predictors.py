import abc
from typing import Iterator, Iterable, List, Any, Dict, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer, PreTrainedModel


class Predictor(abc.ABC):

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = eval_batch_size

    @abc.abstractmethod
    def predict(self, dataset: Dataset) -> Tuple[List[str], List[str]]:
        pass


class FinetunedQAHeadPredictor(Predictor):

    def predict(self, dataset: Dataset) -> Tuple[List[str], List[str]]:
        from qa_utils import QAProcessor
        qa_processor = QAProcessor(self.model, self.tokenizer)
        predictions, references = qa_processor.evaluate_model_for_qa(dataset, self.batch_size)
        return predictions, references


class FinetunedClassifier(Predictor):

    def predict(self, dataset: Dataset) -> Tuple[List[str], List[str]]:
        predicted = []
        actual = []
        for batch in dataset.map(batched=True, batch_size=self.batch_size):
            batch_predictions = [self.model.config.label_map[i] for i in self.model(**batch).argmax(-1)]
            batch_actual = [self.model.config.label_map[i] for i in batch["labels"]]

            predicted.extend(batch_predictions)
            actual.extend(batch_actual)

        return predicted, actual


class GenerativePredictor(Predictor, abc.ABC):
    # TODO: verbalizers

    def predict(self, dataset: Dataset) -> Iterator[str]:
        pass


class LMPredictor(Predictor, abc.ABC):
    # TODO: verbalizers

    def predict(self, dataset: Dataset) -> Iterator[str]:
        pass
