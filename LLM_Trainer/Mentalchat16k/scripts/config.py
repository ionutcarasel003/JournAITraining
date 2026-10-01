from dataclasses import dataclass
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # MentalChat root

BASE_ADAPTER_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "EmpatheticDialogues", "output"))

@dataclass
class TrainingConfig:
    base_model_name = "HuggingFaceH4/zephyr-7b-beta"
    base_adapter_path = BASE_ADAPTER_PATH
    
    output_dir = os.path.join(PROJECT_ROOT, "output")
    data_dir = os.path.join(PROJECT_ROOT, "dataset")
    
    max_length = 512

    # Sequential Training Parameters
    learning_rate = 5e-6
    
    num_train_epochs = 5
    train_batch_size = 2
    eval_batch_size = 8
    gradient_accumulation_steps = 16 
    
    warmup_steps = 200
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
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

    load_in_4bit = False
    bnb_4bit_compute_dtype = "float16"
    
    # Dataset subsetting
    use_subset = False
    subset_size = 12000

    use_validation_subset = True
    validation_subset_size = 100
    
    early_stopping_patience = 3
    early_stopping_threshold = 0.001
    
    dataloader_num_workers = 0
    dataloader_prefetch_factor = None
