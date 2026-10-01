import pandas as pd
import numpy as np

def load_and_prepare_datasets(train_path, val_path, test_path):
    # Load TSV-like text files with semicolon separator and two columns: text;label
    df_train = pd.read_csv(train_path, sep=';', names=['text', 'label'], header=None)
    df_val = pd.read_csv(val_path, sep=';', names=['text', 'label'], header=None)
    df_test = pd.read_csv(test_path, sep=';', names=['text', 'label'], header=None)

    # Get all unique labels from training set
    labels = sorted(df_train['label'].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(labels)

    # Convert single labels to multi-label binary vectors
    def single_to_multilabel(df):
        # Create binary vectors for each sample
        binary_labels = []
        for label in df['label']:
            binary_vector = [0.0] * num_labels
            if label in label2id:
                binary_vector[label2id[label]] = 1.0
            binary_labels.append(binary_vector)
        
        df = df.copy()
        df['label'] = binary_labels
        return df

    # Apply conversion to all datasets
    df_train = single_to_multilabel(df_train)
    df_val = single_to_multilabel(df_val)
    df_test = single_to_multilabel(df_test)

    return df_train, df_val, df_test, label2id, id2label
