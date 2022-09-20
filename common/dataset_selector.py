from typing import Union

from datasets import Dataset
from datasets import load_dataset


def get_dataset(task: str, dataset_type: str, split: str, firstn: Union[bool, int] = False) -> Dataset:
    if task == "QA":
        if dataset_type == "ID":
            dataset = load_dataset("squad")[split]
        elif dataset_type == "OOD":
            dataset = load_dataset("adversarial_qa", "adversarialQA")[split]
        else:
            raise ValueError()
    elif task == "NLI":
        if dataset_type == "ID":
            dataset = load_dataset("multi_nli")[split]
        elif dataset_type == "OOD":
            # dataset = load_dataset("anli")[split]
            dataset = load_dataset("hans")[split]
        else:
            raise ValueError()
    elif task == "Paraphrasing":
        if dataset_type == "ID":
            dataset = load_dataset("glue", "mrpc")[split]
        elif dataset_type == "OOD":
            dataset = load_dataset("paws", "labeled_final")[split]
        else:
            raise ValueError()
    else:
        raise ValueError()

    if firstn:
        dataset = dataset.select(range(firstn))
    return dataset
