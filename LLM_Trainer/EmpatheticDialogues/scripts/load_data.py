import os
from datasets import load_dataset

from menajera.split_dataset import train_file


def load_data(dataset_path):
    train_file = os.path.join(dataset_path, 'train.jsonl')
    test_file = os.path.join(dataset_path, 'test.jsonl')
    eval_file = os.path.join(dataset_path, 'validation.jsonl')

    dataset = load_dataset(
        "json",
        data_files={"train": train_file,
                    "test": test_file,
                    "validation": eval_file
        }
    )

    return dataset