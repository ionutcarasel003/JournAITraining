from transformers import TrainingArguments, EarlyStoppingCallback, Trainer
import torch
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
    datasets = load_data(config.data_dir)
    tokenizer = load_tokenizer(config.base_model_name)
    if config.use_subset:
        datasets["train"] = datasets["train"].select(range(min(config.subset_size, len(datasets["train"]))))
        if hasattr(config, 'use_validation_subset') and config.use_validation_subset:
            val_size = min(config.validation_subset_size, len(datasets["validation"]))
        else:
            val_size = min(config.subset_size // 4, len(datasets["validation"]))
        datasets["validation"] = datasets["validation"].select(range(val_size))
        print(f"🔬 Mod test: {len(datasets['train'])} train, {len(datasets['validation'])} validation")
    tokenized_train = datasets["train"].map(
        lambda x: tokenize_function(x, tokenizer, config.max_length),
        remove_columns=datasets["train"].column_names,
    )
    
    tokenized_val = datasets["validation"].map(
        lambda x: tokenize_function(x, tokenizer, config.max_length),
        remove_columns=datasets["validation"].column_names,
    )

    model = load_model()
    model = apply_lora(model)

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

    # Re-enable evaluation with StableTrainer
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

    print(f"\nTraining Configuration:")
    print(f"   - Device: Apple M4 Max GPU (MPS) ")
    print(f"   - Training: MPS (fast) ⚡")
    print(f"   - Evaluation: CPU (stable) ")
    print(f"   - Early stopping: patience={config.early_stopping_patience}")
    print(f"   - Batch size: {config.train_batch_size}")
    print(f"   - Eval batch size: {config.eval_batch_size}")
    print(f"   - Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Gradient clipping: max_grad_norm=0.5")
    print(f"   - Precision: float16")
    print(f"   - Optimizer: adamw_torch")
    print(f"   - Total epochs: {config.num_train_epochs}")
    print(f"   - Warmup steps: {config.warmup_steps}\n")
    print(f"\n Solution: Evaluation runs on CPU to avoid MPS float16 NaN issues")
    trainer.train()
    model.save_pretrained(config.output_dir)

    print("\n" + "="*50)
    print("Generare grafic cu metrici...")
    print("="*50)
    plot_training_metrics(output_dir=config.output_dir, save_path="training_loss_plot.png")

if __name__ == "__main__":
    main()
    