def format_example(example):
    prompt = (
        f"{example['system']}\n\n"
        f"### User:\n{example['user']}\n\n"
        f"### Assistant:\n{example['assistant']}"
    )
    return prompt

def tokenize_function(example, tokenizer, max_length):
    # Format the full text
    full_text = format_example(example)
    
    # Format just the prompt (without assistant response)
    prompt_text = (
        f"{example['system']}\n\n"
        f"### User:\n{example['user']}\n\n"
        f"### Assistant:\n"
    )
    
    # Tokenize WITHOUT padding first to get actual lengths
    tokenized_no_pad = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False
    )
    
    prompt_tokenized_no_pad = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding=False
    )
    
    # Now tokenize with padding for model input
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    
    # Calculate actual content lengths
    actual_length = len(tokenized_no_pad["input_ids"])
    prompt_length = len(prompt_tokenized_no_pad["input_ids"])
    
    # Find where actual content starts (after left-padding)
    padding_length = max_length - actual_length
    
    # Create labels: start with all -100 (masked)
    labels = [-100] * max_length
    
    # Only unmask the assistant's response tokens
    # Assistant response starts at [padding_length + prompt_length : padding_length + actual_length]
    assistant_start = padding_length + prompt_length
    assistant_end = padding_length + actual_length
    
    # Copy the assistant response tokens to labels
    for i in range(assistant_start, assistant_end):
        labels[i] = tokenized["input_ids"][i]
    
    tokenized["labels"] = labels
    return tokenized
