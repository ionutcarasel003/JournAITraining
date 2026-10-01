import os
from dataclasses import dataclass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # HarmPrevention root
LLM_ROOT = os.path.dirname(PROJECT_ROOT)  # LLM root

# Path to the base model and SFT adapter from MentalChat
BASE_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
SFT_ADAPTER_PATH = os.path.join(LLM_ROOT, "EmpatheticDialogues", "output")

@dataclass
class HarmPreventionDPOConfigEd:
    # Model and paths
    base_model_name = BASE_MODEL_NAME
    sft_adapter_path = SFT_ADAPTER_PATH
    output_dir = os.path.join(PROJECT_ROOT, "empathetic_dialogues_dpo")

    # Dataset settings
    dataset_name = "Anthropic/hh-rlhf"
    dataset_subset = "harmless-base" # Focus specifically on harm prevention

    # Context constraints
    max_length = 512
    max_prompt_length = 256

    # DPO Hyperparameters
    beta = 0.1  # Implicit reward scale factor (standard for DPO is 0.1)
    learning_rate = 5e-7  # DPO requires a very small learning rate
    weight_decay = 0.01

    # Optimization/batching (optimized for local macOS MPS execution)
    num_train_epochs = 1
    train_batch_size = 1
    eval_batch_size = 1
    gradient_accumulation_steps = 16  # Large accumulation to stabilize gradients

    warmup_steps = 50
    logging_steps = 10
    save_steps = 50
    eval_steps = 50
    eval_strategy = "steps"

    # LoRA Hyperparameters (consistent with MentalChat)
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.1
    target_modules = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    # Hardware/Execution settings
    # MPS is default on Apple Silicon, else CPU
    device = "mps"

    # Dataset Subsetting (for fast iteration/debugging or limited resources)
    use_subset = True
    subset_size = 2000  # Subset of harmless-base to speed up training
    use_validation_subset = True
    validation_subset_size = 100
