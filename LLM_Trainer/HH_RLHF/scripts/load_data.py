import re
from datasets import load_dataset
from transformers import AutoTokenizer

def parse_conversation(text):
    """
    Parses Anthropic's HH-RLHF conversation format:
    '\n\nHuman: [text]\n\nAssistant: [text]'
    Returns a list of dictionaries with 'role' and 'content'.
    """
    # Split text into human and assistant segments
    parts = re.split(r'\n\n(Human|Assistant): ', text)
    
    messages = []
    current_role = None
    
    # The first element before any match is usually empty or whitespace
    # We iterate starting from the first matched role
    for i in range(1, len(parts), 2):
        role_marker = parts[i]
        content = parts[i+1].strip()
        
        if role_marker == "Human":
            current_role = "user"
        elif role_marker == "Assistant":
            current_role = "assistant"
            
        if current_role and content:
            messages.append({"role": current_role, "content": content})
            
    return messages

def prepare_dpo_datasets(config, tokenizer):
    """
    Loads Anthropic/hh-rlhf, formats and prepares train/validation subsets.
    """
    print(f"Loading dataset {config.dataset_name} (subset: {config.dataset_subset})...")
    
    # Load from HF Datasets
    raw_datasets = load_dataset(config.dataset_name, data_dir=config.dataset_subset)
    
    system_prompt = (
        "You are an empathetic AI assistant trained to provide mental health support. "
        "The assistant gives helpful, comprehensive, and appropriate answers to the user's questions."
    )
    
    def process_function(example):
        chosen_msgs = parse_conversation(example["chosen"])
        rejected_msgs = parse_conversation(example["rejected"])
        
        if not chosen_msgs or not rejected_msgs:
            return {"prompt": "", "chosen": "", "rejected": ""}
            
        # The prompt represents the history before the last assistant turn
        # Both chosen and rejected should share the exact same prompt history
        prompt_msgs = chosen_msgs[:-1]
        
        # Prepend the system prompt for alignment with SFT
        full_prompt_msgs = [{"role": "system", "content": system_prompt}] + prompt_msgs
        
        # Apply tokenizer's chat template to the prompt
        # add_generation_prompt=True adds the "<|assistant|>\n" token at the end
        prompt_formatted = tokenizer.apply_chat_template(
            full_prompt_msgs, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # The chosen and rejected completions
        # Ensure they end with the tokenizer's eos_token
        chosen_response = chosen_msgs[-1]["content"] + tokenizer.eos_token
        rejected_response = rejected_msgs[-1]["content"] + tokenizer.eos_token
        
        return {
            "prompt": prompt_formatted,
            "chosen": chosen_response,
            "rejected": rejected_response
        }

    # Apply preprocessing mapping
    print("Preprocessing and formatting dataset turns...")
    
    # Run mapping on splits
    processed_datasets = raw_datasets.map(
        process_function,
        remove_columns=raw_datasets["train"].column_names,
        desc="Formatting chat template"
    )
    
    # Filter out empty examples (if parsing failed or inputs were malformed)
    processed_datasets = processed_datasets.filter(
        lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0 and len(x["rejected"]) > 0
    )
    
    # Create subsets for resource-constrained training
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]
    
    if config.use_subset:
        print(f"Subsetting train dataset to {config.subset_size} samples...")
        train_dataset = train_dataset.select(range(min(config.subset_size, len(train_dataset))))
        
    if config.use_validation_subset:
        print(f"Subsetting validation dataset to {config.validation_subset_size} samples...")
        eval_dataset = eval_dataset.select(range(min(config.validation_subset_size, len(eval_dataset))))
        
    print(f"Final dataset sizes - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset
