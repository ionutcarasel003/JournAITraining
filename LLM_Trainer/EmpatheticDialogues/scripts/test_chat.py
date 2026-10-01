import sys
import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer

# Ensure we can import from local modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

try:
    from config import TrainingConfig
    from model_setup import load_model, TrainingConfig  # Import config via model_setup or directly
    from tokenizer import load_tokenizer
except ImportError:
    # Check if we are running from root
    sys.path.append(os.path.join(SCRIPT_DIR, '..'))
    from training_scripts.config import TrainingConfig
    from training_scripts.model_setup import load_model
    from training_scripts.tokenizer import load_tokenizer

def load_trained_model():
    print("Loading base model...")
    # Load base model (uses configuration from model_setup.py)
    model = load_model(force_cpu=False) 
    
    # Ensure caching is enabled for faster inference generated content
    model.config.use_cache = True
    
    # Path to the saved adapter
    adapter_path = TrainingConfig.output_dir
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    try:
        # Load the LoRA adapter
        model = PeftModel.from_pretrained(model, adapter_path)
    except Exception as e:
        print(f"⚠️ Error loading adapter: {e}")
        print(f"Attempting to check if {adapter_path} exists...")
        if not os.path.exists(adapter_path):
            print(f"❌ Directory {adapter_path} does not exist. Please train the model first.")
            sys.exit(1)
        else:
            # Fallback for some Peft versions or path issues
            print("Trying to load with is_trainable=False...")
    
    return model

def chat_loop():
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(TrainingConfig.base_model_name)
    
    model = load_trained_model()
    model.eval()
    
    print(f"Inference device: {model.device}")

    print("\n" + "="*50)
    print("🤖 Empathetic Chatbot")
    print("   (Type 'quit', 'exit', or 'q' to stop)")
    print("="*50)

    # Default persona from training data style if applicable
    system_prompt = "You are an empathetic assistant skilled in emotional support. You listen actively and respond with warmth and understanding."
    print(f"System Persona: {system_prompt}\n")

    while True:
        try:
            user_input = input("\033[1;34mYou:\033[0m ") # Blue "You:"
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            
            if not user_input.strip():
                continue
            
            # Format prompt matching training data
            # {system}\n\n### User:\n{user}\n\n### Assistant:\n
            prompt = f"{system_prompt}\n\n### User:\n{user_input}\n\n### Assistant:\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's part
            if "### Assistant:\n" in full_response:
                response = full_response.split("### Assistant:\n")[-1].strip()
            else:
                # If the tag is missing (rare), try to remove the prompt manually
                # This is a fallback
                response = full_response[len(prompt):].strip()
                # If prompt removal fails clean up artifacts
                response = response.replace(system_prompt, "").replace("### User:", "").replace("### Assistant:", "").strip()

            print(f"\033[1;32mAssistant:\033[0m {response}\n") # Green "Assistant:"
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during generation: {e}")

    print("\nGoodbye! 👋")

if __name__ == "__main__":
    chat_loop()
