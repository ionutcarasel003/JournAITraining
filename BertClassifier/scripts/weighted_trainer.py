from typing import Optional, Tuple

import torch
from torch import nn
from transformers import Trainer


class WeightedTrainer(Trainer):
    #Trainer that applies BCEWithLogitsLoss with per-class pos_weight for multi-label.
    #pos_weight[k] > 1 increases the loss for positive examples of class k (rare classes).

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # shape [num_labels]

    def compute_loss(self,model,inputs,
        return_outputs: bool = False,num_items_in_batch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[object]]:
        labels = inputs.get("labels")
        # Forward pass (exclude labels from kwargs)
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits

        loss_fct = nn.BCEWithLogitsLoss(
            pos_weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits, labels.float())

        return (loss, outputs) if return_outputs else loss


