# Preprocessing functions for validation dataset from the HuggingFace Jupyter notebook with original comments
import collections
from typing import Tuple, List, Dict
import inspect
import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedModel


class QAProcessor:

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_model_for_qa(self, dataset_eval: Dataset, batch_size: int = 32) -> Tuple[List[str], List[str]]:
        """Model evaluation on specific dataset
        Calls previous functions and evaluate the dataset on the model for exact match and F1

        Args:
            dataset_eval (Dataset): validation dataset
            batch_size

        Returns:
            Tuple[Dict[str, float], Dataset]: dictionary of metrics and dataset with predictions
        """

        validation_features = dataset_eval.map(
            self.prepare_validation_features,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset_eval.column_names
        )
        # validation_features.set_format(type="pt", columns=list(validation_features.features.keys()))

        raw_predictions = self.collect_predictions(validation_features, batch_size)

        final_predictions = self.postprocess_qa_predictions(dataset_eval, validation_features, raw_predictions)

        # formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
        # references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset_eval]

        keys_ordered = list(final_predictions.keys())
        predictions_ordered = [final_predictions[k] for k in keys_ordered]
        references_ordered = [sample["answers"]["text"][0] for k in keys_ordered for sample in dataset_eval if sample["id"] == k]

        return predictions_ordered, references_ordered

    def collect_predictions(self, dataset_feats: Dataset, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_logits_l = []
        end_logits_l = []
        for batch_offset in tqdm(range(0, len(dataset_feats), batch_size), desc="Collecting QA predictions"):
            batch = dataset_feats[batch_offset: batch_offset + batch_size]

            model_args = inspect.getfullargspec(self.model.forward).args
            with torch.no_grad():
                outputs = self.model(**{k: torch.tensor(v) for k, v in batch.items() if k in model_args})

            start_logits_l.append(outputs.start_logits)
            end_logits_l.append(outputs.end_logits)

        return torch.vstack(start_logits_l), torch.vstack(end_logits_l)

    def prepare_validation_features(self, examples) -> BatchEncoding:
        """Function for processing batches of validation dataset

        Args:
            examples (datasets.arrow_dataset.Batch): batches of the dataset

        Returns:
            BatchEncoding: prepared batches of the dataset
        """

        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        pad_on_right = self.tokenizer.padding_side == "right"
        max_length = 384
        doc_stride = 128

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This
        # results in one example possible giving several features when a context is long, each of those features
        # having a context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            # return_tensors="pt"
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example
            # (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
        #
        #     # # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        #     # # position is part of the context or not.
        #     tokenized_examples["offset_mapping"][i] = [
        #         (o if sequence_ids[k] == context_index else None)
        #         for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        #     ]

        return tokenized_examples

    # Postprocessing function from the HuggingFace Jupyter notebook
    # with original comments
    def postprocess_qa_predictions(self,
                                   examples: Dataset, features: Dataset,
                                   raw_predictions: Tuple[torch.Tensor, torch.Tensor],
                                   n_best_size: int = 20,
                                   max_answer_length: int = 30) -> Dict[str, str]:
        """Postprocessing of predictions of the QA model

        Args:
            examples (Dataset): validation dataset
            features (Dataset): batched and processed dataset
            raw_predictions (Tuple[List[float], List[float]]): arrays of start and end logits
            n_best_size (int, optional): number of best logits to search. Defaults to 20.
            max_answer_length (int, optional): length of the asnwer. Defaults to 30.

        Returns:
            collections.OrderedDict: _description_
        """
        if len(raw_predictions) == 2:
            all_start_logits, all_end_logits = raw_predictions
        else:
            raise ValueError()
            # all_start_logits, all_end_logits, _ = raw_predictions

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()

        # Logging.
        print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_score = None  # Only used if squad_v2 is True.
            valid_answers = []

            context = example["context"]
            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the
                # original context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(self.tokenizer.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = start_logits.argsort(descending=True).tolist()
                end_indexes = end_logits.argsort(descending=True).tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or
                        # correspond to part of the input_ids that are not in the context.
                        if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue
                        if len(offset_mapping[start_index]) == 0 or len(offset_mapping[end_index]) == 0:
                            continue

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append(
                            {
                                "score": start_logits[start_index] + end_logits[end_index],
                                "text": context[start_char: end_char]
                            }
                        )

            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            else:
                # In the very rare edge case we have not a single non-null prediction,
                # we create a fake prediction to avoid failure.
                best_answer = {"text": "", "score": 0.0}

            # Let's pick our final answer: the best one or the null answer (only for squad_v2)
            squad_v2 = False
            if not squad_v2:
                predictions[example["id"]] = best_answer["text"]
            else:
                answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
                predictions[example["id"]] = answer

        return predictions
