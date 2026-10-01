from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from load_data import load_and_prepare_datasets
from preprocess import tokenize_and_wrap
import metrics as metrics_module

from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent

    train_path = repo_root / "dataset" / "train_augmented.txt"
    val_path = repo_root / "dataset" / "val_augmented.txt"
    test_path = repo_root / "dataset" / "test_augmented.txt"

    _, _, df_test, label2id, id2label = load_and_prepare_datasets(
        str(train_path), str(val_path), str(test_path)
    )
    labels = [id2label[i] for i in range(len(label2id))]

    model_dir = repo_root / "augmented_model"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    test_dataset = tokenize_and_wrap(tokenizer, df_test)

    trainer = Trainer(model=model)

    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    y_true = predictions.label_ids

    probs = 1 / (1 + np.exp(-logits))
    if metrics_module.THRESHOLDS is not None:
        thr = np.array(metrics_module.THRESHOLDS, dtype=float)
        if thr.ndim == 1:
            thr = thr.reshape(1, -1)
        y_pred = (probs >= thr).astype(int)
    else:
        y_pred = (probs >= 0.5).astype(int)


    subset_acc = (y_pred == y_true).all(axis=1).mean()

    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true.flatten(), y_pred.flatten(), average="micro", zero_division=0
    )

    print("Evaluation Metrics on Test Set:")
    print(f"Subset accuracy: {subset_acc:.4f}")
    print(f"Precision (micro): {p_micro:.4f}  Recall (micro): {r_micro:.4f}  F1 (micro): {f1_micro:.4f}")
    print()

    p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    for i, lab in enumerate(labels):
        print(f"{lab:>10s}: P={p_c[i]:.3f} R={r_c[i]:.3f} F1={f1_c[i]:.3f} Support={int(sup_c[i])}")


if __name__ == "__main__":
    main()
