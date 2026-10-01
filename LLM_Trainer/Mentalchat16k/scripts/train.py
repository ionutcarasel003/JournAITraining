from transformers import TrainingArguments, EarlyStoppingCallback, Trainer
import torch
import os
from peft import PeftModel
from load_data import load_data
from preprocessing import tokenize_function
from tokenizer import load_tokenizer
from model_setup import load_model
from lora_setup import apply_lora
from config import TrainingConfig
from plot_metrics import plot_training_metrics
from stable_trainer import StableTrainer


def main():
    config = TrainingConfig()
    
    print("\n" + "="*60)
    print(" SEQUENTIAL TRAINING INITIATION")
    print("="*60)
    print(f"Dataset Directory: {config.data_dir}")
    print(f"Output Directory:  {config.output_dir}")
    print(f"Base Model:        {config.base_model_name}")
    print(f"Pre-trained Adapter: {config.base_adapter_path}")
    print("="*60 + "\n")

    # 1. Load Data
    print(" Loading MentalChat Dataset...")
    datasets = load_data(config.data_dir)
    tokenizer = load_tokenizer(config.base_model_name)

    # Subset usage (defaults to False for full training)
    if config.use_subset:
        print(f" Using subset of {config.subset_size} samples")
        datasets["train"] = datasets["train"].select(range(min(config.subset_size, len(datasets["train"]))))
        
    # Validation subset
    if hasattr(config, 'use_validation_subset') and config.use_validation_subset:
        val_size = min(config.validation_subset_size, len(datasets["validation"]))
        datasets["validation"] = datasets["validation"].select(range(val_size))
        print(f" Fast Validation: using {len(datasets['validation'])} samples")

    # 2. Tokenize
    print("Tokenizing datasets...")
    tokenized_train = datasets["train"].map(
        lambda x: tokenize_function(x, tokenizer, config.max_length),
        remove_columns=datasets["train"].column_names,
    )
    
    tokenized_val = datasets["validation"].map(
        lambda x: tokenize_function(x, tokenizer, config.max_length),
        remove_columns=datasets["validation"].column_names,
    )

    # 3. Load Model (Sequential Logic)
    print("Loading Base Model...")
    model = load_model()
    
    # Check for existing adapter to continue training
    if config.base_adapter_path and os.path.exists(os.path.join(config.base_adapter_path, 'adapter_model.safetensors')):
        print(f"\n🔄 LOADING PRE-TRAINED ADAPTER: {config.base_adapter_path}")
        print("   Resuming training from EmpatheticDialogues checkpoint...")
        
        # Load the adapter and mark it as trainable
        model = PeftModel.from_pretrained(
            model, 
            config.base_adapter_path, 
            is_trainable=True
        )
        print(" Adapter loaded successfully and set to TRAINABLE mode.")
        model.print_trainable_parameters()
    else:
        print("\n No pre-trained adapter found (or path invalid).")
        print("   Initializing NEW LoRA adapter...")
        model = apply_lora(model)

    # 4. Training Arguments
    training_args = TrainingArguments(
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
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        fp16=False,
        bf16=False,
        report_to="none",
        prediction_loss_only=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=config.dataloader_num_workers,
        save_total_limit=3,
        eval_accumulation_steps=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_grad_norm=0.5,
        greater_is_better=False,
        eval_on_start=False,
        logging_nan_inf_filter=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    # 5. Initialize Trainer
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_threshold=config.early_stopping_threshold
    )

    trainer = StableTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        callbacks=[early_stopping],
    )

    print(f"\n Training Configuration:")
    print(f"   - Starting Loss should be low (< 1.0) if adapter works.")
    print(f"   - Evaluation running on CPU.")
    
    # 6. Train
    trainer.train()
    
    # 7. Save
    print("\n Saving fine-tuned MentalChat model...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # 8. Plot
    print("\n Generating Metrics Plot...")
    plot_training_metrics(output_dir=config.output_dir, save_path="mentalchat_training_plot.png")
    print("\n DONE! Sequential training complete.")

if __name__ == "__main__":
    main()
