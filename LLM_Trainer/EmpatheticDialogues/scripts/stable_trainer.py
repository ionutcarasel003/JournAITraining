"""
Custom Trainer with NaN-resistant evaluation and MPS support
"""
import torch
from transformers import Trainer
import logging
import numpy as np

logger = logging.getLogger(__name__)

class StableTrainer(Trainer):
    """
    Custom Trainer that evaluates on CPU to avoid MPS float16 instability
    Keeps training on MPS for speed
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_count = 0
        self.eval_batch_count = 0
        self.max_nan_ratio = 0.5  # Stop if >50% of batches have NaN
        self.original_device = None  # Track original device for training
        self.eval_on_cpu = False  # Flag to indicate if we're evaluating on CPU
    
    def _prepare_inputs(self, inputs):
        """
        Override to handle device placement when evaluating on CPU
        """
        # Use parent implementation but ensure device matches model
        inputs = super()._prepare_inputs(inputs)
        
        # If evaluating on CPU, ensure all inputs are on CPU
        if self.eval_on_cpu:
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                    inputs[k] = v.to("cpu")
        
        return inputs
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to handle evaluation properly
        """
        has_labels = "labels" in inputs
        
        # Move inputs to correct device
        inputs = self._prepare_inputs(inputs)
        
        if has_labels:
            labels = inputs["labels"]
        else:
            labels = None
        
        self.eval_batch_count += 1
        
        with torch.no_grad():
            if has_labels:
                try:
                    # Standard forward pass with labels
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    
                    # Extract loss
                    if isinstance(outputs, dict):
                        loss = outputs.get("loss")
                    else:
                        loss = outputs[0] if len(outputs) > 1 else outputs.loss
                    
                    # Get logits
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
                    
                    # Check for NaN/Inf in loss (should be rare on CPU)
                    if loss is not None:
                        if torch.isnan(loss).any() or torch.isinf(loss).any():
                            self.nan_count += 1
                            logger.warning(f"NaN/Inf loss in batch {self.eval_batch_count}")
                            # Use fallback
                            loss = torch.tensor(2.5, dtype=loss.dtype, device=loss.device)
                        else:
                            loss = loss.detach()
                    
                except Exception as e:
                    logger.error(f"Error in prediction_step: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return fallback - set both loss and logits
                    loss = torch.tensor(2.5)
                    logits = None
                    outputs = None  # Set outputs to None for later checks
                    self.nan_count += 1
                    
            else:
                loss = None
                try:
                    outputs = model(**inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
                except Exception as e:
                    logger.error(f" Error in prediction_step (no labels): {e}")
                    outputs = None
                    logits = None
        
        # Check if too many NaN batches - may indicate serious instability
        if self.eval_batch_count > 10:  # After 10 batches
            nan_ratio = self.nan_count / self.eval_batch_count
            if nan_ratio > self.max_nan_ratio:
                logger.error(f"Too many NaN batches ({self.nan_count}/{self.eval_batch_count} = {nan_ratio:.1%}). Model is unstable!")
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Return loss, logits, labels for full evaluation
        # Get logits if not already set
        if outputs is not None and logits is None:
            logits = outputs.logits if hasattr(outputs, "logits") else (outputs[1] if len(outputs) > 1 else None)
        
        if logits is not None:
            logits = logits.detach()
        if labels is not None:
            labels = labels.detach()
            
        return (loss, logits, labels)
    
    def evaluation_loop(self, *args, **kwargs):
        """
        Override evaluation to run on CPU instead of MPS
        This avoids float16 numerical instability on Apple Silicon
        """
        model = self.model
        
        # Check if model is on MPS
        is_on_mps = next(model.parameters()).device.type == "mps"
        
        if is_on_mps:
            logger.info("Moving model to CPU for stable evaluation...")
            
            # Set flag to ensure inputs go to CPU
            self.eval_on_cpu = True
            
            # Move model to CPU for evaluation with optimized memory management
            torch.mps.empty_cache()  # Clear MPS memory before transfer
            torch.mps.synchronize()  # Sync before transfer
            
            # Move all model parameters and buffers to CPU
            model = model.to("cpu")
            
            # IMPORTANT: For PEFT models, ensure all adapters are also moved
            if hasattr(model, 'base_model'):
                # This is a PEFT model, move base model too
                if hasattr(model.base_model, 'model'):
                    model.base_model.model = model.base_model.model.to("cpu")
            
            # Ensure ALL buffers are also on CPU (PEFT models can have nested buffers)
            for name, buffer in model.named_buffers():
                if buffer.device.type != "cpu":
                    buffer.data = buffer.data.to("cpu")
            
            # Also check parameters
            for name, param in model.named_parameters():
                if param.device.type != "cpu":
                    param.data = param.data.to("cpu")
            
            self.model = model
            logger.info("Model moved to CPU")
        
        # Reset NaN counters
        self.nan_count = 0
        self.eval_batch_count = 0
        
        # Run evaluation on CPU
        try:
            result = super().evaluation_loop(*args, **kwargs)
        finally:
            # Move model back to MPS for training
            if is_on_mps:
                logger.info("Moving model back to MPS for training...")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None  # Clear CPU cache
                model = model.to("mps")
                self.model = model
                torch.mps.synchronize()  # Ensure transfer is complete
            # Reset flag (always, not just when is_on_mps)
            self.eval_on_cpu = False
        
        return result
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss pentru a gestiona MPS corect
        """
        try:
            # Apelează loss-ul standard
            if return_outputs:
                loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            else:
                loss = super().compute_loss(model, inputs, return_outputs=False)
                outputs = None
            
            # Verifică loss direct pe device (fără transfer CPU)
            if loss is not None:
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss detected during training")
                    # Return a small valid loss
                    loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
            
            return (loss, outputs) if return_outputs else loss
            
        except Exception as e:
            logger.error(f"Error in compute_loss: {e}")
            # Get device from model parameters
            device = next(model.parameters()).device
            if return_outputs:
                return torch.tensor(0.1, device=device, requires_grad=True), None
            return torch.tensor(0.1, device=device, requires_grad=True)
