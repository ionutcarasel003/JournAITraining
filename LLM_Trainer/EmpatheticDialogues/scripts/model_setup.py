import torch
from torch.xpu import device
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

try:
    from .config import TrainingConfig
except ImportError:
    from config import TrainingConfig

def load_model(force_cpu=False):
    """
    Load the model on the appropriate device.
    
    Args:
        force_cpu: If True, load model on CPU regardless of MPS availability
    """
    # Only use quantization config if 4-bit loading is enabled
    if TrainingConfig.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=TrainingConfig.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, TrainingConfig.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            TrainingConfig.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # Apple Silicon (M4 Max) - folosește float16 (MPS nu suportă bfloat16)
        if torch.backends.mps.is_available() and not force_cpu:
            print("🚀 Using Apple Silicon GPU (MPS)")
            model = AutoModelForCausalLM.from_pretrained(
                TrainingConfig.base_model_name,
                trust_remote_code=True,
                dtype=torch.float16  # MPS folosește float16 nativ
            )
            model = model.to("mps")  # Mutăm explicit pe MPS
            print("   ⚠️  Note: Evaluation will run on CPU due to MPS numerical instability")
        else:
            if force_cpu:
                print("💻 Using CPU (forced)")
            else:
                print("⚠️  MPS nu e disponibil, folosesc CPU")
            model = AutoModelForCausalLM.from_pretrained(
                TrainingConfig.base_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            model = model.to("cpu")

    model.config.use_cache = False
    return model
