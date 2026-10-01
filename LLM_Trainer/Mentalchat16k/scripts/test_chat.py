import sys
import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

try:
    from config import TrainingConfig
    from model_setup import load_model
    from tokenizer import load_tokenizer
except ImportError:
    sys.path.append(os.path.join(SCRIPT_DIR, '..'))
    from config import TrainingConfig
    from model_setup import load_model
    from tokenizer import load_tokenizer

def load_trained_model():
    print("Loading base model...")
    model = load_model(force_cpu=False)
    model.config.use_cache = True

    adapter_path = TrainingConfig.output_dir
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print(f"Checking {adapter_path}...")
        if not os.path.exists(adapter_path):
            print(f" Directory {adapter_path} does not exist. Please train the model first.")
            sys.exit(1)
            
    return model

def chat_loop():
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(TrainingConfig.base_model_name)
    
    model = load_trained_model()
    model.eval()
    
    print(f"Inference device: {model.device}")

    print("\n" + "="*50)
    print(" MentalChat AI Assistant")
    print("   (Type 'quit', 'exit', or 'q' to stop)")
    print("="*50)

    system_prompt = "You are an empathetic AI assistant trained to provide mental health support. The assistant gives helpful, comprehensive, and appropriate answers to the user's questions."
    print(f"System Context: {system_prompt}\n")

    while True:
        try:
            user_input = input("\033[1;36mPatient:\033[0m ") # Cyan "Patient:"
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            
            if not user_input.strip():
                continue
            
            # Format prompt matching training data
            # Use the tokenizer's chat template if available or fallback to manual formatting
            # Our training data used:
            # {"system": "...", "user": "...", "assistant": "..."}
            # Zephyr template expects system, user, assistant messages.
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
            
            # Apply chat template
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            print(f"\033[1;32mMentalChat:\033[0m {response.strip()}\n") # Green "MentalChat:"
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during generation: {e}")

    print("\nTake care!")

if __name__ == "__main__":
    chat_loop()
