from dataclasses import dataclass
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

@dataclass
class TrainingConfig:
    base_model_name = "HuggingFaceH4/zephyr-7b-beta"
    output_dir = os.path.join(PROJECT_ROOT, "output")

    data_dir = os.path.join(PROJECT_ROOT, "DS_clean")
    max_length = 512

    num_train_epochs = 3
    train_batch_size = 2
    eval_batch_size = 8
    gradient_accumulation_steps = 16
    learning_rate = 1e-5
    warmup_steps = 500
    logging_steps = 20
    weight_decay = 0.1
    save_steps = 200
    eval_strategy = "steps"
    eval_steps = 100
    fp16 = False
    bf16 = False

    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.1
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Attention + MLP

    load_in_4bit = False
    bnb_4bit_compute_dtype = "float16"

    use_subset = True
    subset_size = 10000

    use_validation_subset = True
    validation_subset_size = 50

    early_stopping_patience = 3
    early_stopping_threshold = 0.01

    dataloader_num_workers = 0
    dataloader_prefetch_factor = None