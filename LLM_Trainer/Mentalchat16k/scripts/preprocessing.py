def format_example(example):
    prompt = (
        f"{example['system']}\n\n"
        f"### User:\n{example['user']}\n\n"
        f"### Assistant:\n{example['assistant']}"
    )
    return prompt

def tokenize_function(example, tokenizer, max_length):

    full_text = format_example(example)
    

    prompt_text = (
        f"{example['system']}\n\n"
        f"### User:\n{example['user']}\n\n"
        f"### Assistant:\n"
    )

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

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    actual_length = len(tokenized_no_pad["input_ids"])
    prompt_length = len(prompt_tokenized_no_pad["input_ids"])

    # Create labels: start with all -100 (masked)
    labels = [-100] * max_length
    
    # Only unmask the assistant's response tokens
    # With right padding, assistant response starts at prompt_length and ends at actual_length
    assistant_start = min(prompt_length, max_length)
    assistant_end = min(actual_length, max_length)
    
    # Copy the assistant response tokens to labels
    for i in range(assistant_start, assistant_end):
        labels[i] = tokenized["input_ids"][i]
    
    tokenized["labels"] = labels
    return tokenized
