import os
import sys
import torch
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

# Add scripts directory to path to load local configurations
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from config_ed import HarmPreventionDPOConfigEd
from load_data_ed import prepare_dpo_datasets
from tokenizer import load_tokenizer

class StableDPOTrainer(DPOTrainer):
    """
    Subclass of DPOTrainer that handles evaluation on CPU to prevent
    float16 numerical instability (NaNs) on Apple Silicon (MPS).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_on_cpu = False

    def _move_tensors_to_cpu(self, d):
        if isinstance(d, dict):
            return {k: self._move_tensors_to_cpu(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._move_tensors_to_cpu(v) for v in d]
        elif isinstance(d, torch.Tensor):
            return d.to("cpu")
        return d

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if self.eval_on_cpu:
            inputs = self._move_tensors_to_cpu(inputs)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def evaluation_loop(self, *args, **kwargs):
        model = self.model
        is_on_mps = next(model.parameters()).device.type == "mps"

        if is_on_mps:
            print("\n📊 [StableDPOTrainer] Moving model to CPU for stable float16 evaluation...")
            self.eval_on_cpu = True

            # Flush MPS cache and sync
            torch.mps.empty_cache()
            torch.mps.synchronize()

            # Move model to CPU
            model = model.to("cpu")

            # Handle PEFT internals to ensure base model is also moved to CPU
            if hasattr(model, 'base_model'):
                if hasattr(model.base_model, 'model'):
                    model.base_model.model = model.base_model.model.to("cpu")

            # Move buffers and parameters
            for name, buffer in model.named_buffers():
                if buffer.device.type != "cpu":
                    buffer.data = buffer.data.to("cpu")
            for name, param in model.named_parameters():
                if param.device.type != "cpu":
                    param.data = param.data.to("cpu")

            self.model = model
            print("   ✓ Model moved to CPU. Running evaluation loop...")

        try:
            result = super().evaluation_loop(*args, **kwargs)
        finally:
            if is_on_mps:
                print("\n🚀 [StableDPOTrainer] Moving model back to MPS for training...")
                model = model.to("mps")
                self.model = model
                torch.mps.synchronize()
            self.eval_on_cpu = False

        return result

def main():
    config = HarmPreventionDPOConfigEd()

    print("\n" + "="*60)
    print("🚀 DPO HARM PREVENTION TRAINING INITIATION")
    print("="*60)
    print(f"Base Model:       {config.base_model_name}")
    print(f"SFT Adapter:      {config.sft_adapter_path}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Dataset:          {config.dataset_name} ({config.dataset_subset})")
    print("="*60 + "\n")

    # 1. Load Tokenizer
    print("Tokenizer loading...")
    tokenizer = load_tokenizer(config.base_model_name)

    # 2. Load Datasets
    train_dataset, eval_dataset = prepare_dpo_datasets(config, tokenizer)

    # 3. Load Model and Merge SFT Adapter
    print("🤖 Loading base model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device for training: {device}")

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        trust_remote_code=True
    )

    # Check if SFT adapter exists to merge
    sft_adapter_file = os.path.join(config.sft_adapter_path, 'adapter_model.safetensors')
    if os.path.exists(sft_adapter_file):
        print(f"🔄 Loading SFT adapter from {config.sft_adapter_path}...")
        peft_model = PeftModel.from_pretrained(base_model, config.sft_adapter_path)
        print("Merging SFT adapter into base model weights...")
        model = peft_model.merge_and_unload()
        print("✅ SFT adapter merged successfully!")

        # Clean up PEFT attributes to prevent multiple adapters warnings
        if hasattr(model, "peft_config"):
            delattr(model, "peft_config")
    else:
        print("⚠️ SFT adapter not found. Training DPO directly on base Zephyr model.")
        model = base_model

    if device == "mps":
        model = model.to("mps")

    # 4. Initialize new LoRA for DPO
    print("Initializing new LoRA config for DPO training...")
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    print("Trainable parameters for DPO:")
    model.print_trainable_parameters()

    # 5. Training Arguments (using DPOConfig from trl)
    from trl import DPOConfig

    training_args = DPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        fp16=False,
        bf16=False,
        report_to="none",
        save_total_limit=2,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        # DPO specific parameters passed in config
        beta=config.beta,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
    )

    # 6. Initialize DPOTrainer
    print("Initializing StableDPOTrainer...")
    trainer = StableDPOTrainer(
        model=model,
        ref_model=None,  # PEFT handles reference log probabilities by disabling DPO adapter
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 7. Train
    print("🚀 Starting DPO Fine-tuning...")
    trainer.train()

    # 8. Save DPO Adapter
    print(f"💾 Saving DPO aligned model to {config.output_dir}...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("✅ Training and saving complete!")

if __name__ == "__main__":
    main()
