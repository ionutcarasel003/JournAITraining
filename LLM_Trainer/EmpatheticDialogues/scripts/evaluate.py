import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Use absolute path relative to this file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Use the latest checkpoint
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "output", "checkpoint-939")
BASE_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# Global model and tokenizer to avoid reloading
_model = None
_tokenizer = None

def load_model():
    """Load the model and tokenizer once"""
    global _model, _tokenizer
    
    if _model is None:
        print(f"Loading base model: {BASE_MODEL}")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="cpu",  # Use CPU to avoid offloading issues
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        
        # Load LoRA adapter
        print(f"Loading LoRA adapter from: {MODEL_PATH}")
        _model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        _model.eval()
        
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        _tokenizer.pad_token = _tokenizer.eos_token
    
    return _model, _tokenizer


def generate_response(prompt, emotion=None):
    model, tokenizer = load_model()
    
    # Format prompt using Zephyr chat template
    if emotion:
        system_msg = f"You are an empathetic AI assistant. The user is feeling {emotion}."
    else:
        system_msg = "You are an empathetic AI assistant."
    
    # Use the proper chat template format
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        full_prompt = f"<|system|>\n{system_msg}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip the prompt)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Clean up the response - stop at any special tokens or system prompts
    if '<|' in response:
        response = response.split('<|')[0]
    
    return response.strip()


if __name__ == "__main__":
    test_prompts = [
        ("I'm feeling really anxious about my upcoming exam.", "anxious"),
        ("I just got promoted at work! This is amazing!", "excited"),
        ("My best friend moved away and I feel so alone.", "lonely"),
    ]
    
    for prompt, emotion in test_prompts:
        print("\n" + "="*80)
        print(f"User ({emotion}): {prompt}")
        print("-"*80)
        response = generate_response(prompt, emotion)
        print(f"Assistant: {response}")
        print("="*80)

