from peft import LoraConfig, get_peft_model

try:
    from .config import TrainingConfig
except ImportError:
    from config import TrainingConfig

def apply_lora(model):
    lora_config = LoraConfig(
        r = TrainingConfig.lora_r,
        lora_alpha = TrainingConfig.lora_alpha,
        lora_dropout = TrainingConfig.lora_dropout,
        target_modules = TrainingConfig.target_modules,
        bias = "none",
        task_type = "CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model