import numpy as np

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).astype(np.float32).mean().item()
    return {"accuracy": accuracy}
