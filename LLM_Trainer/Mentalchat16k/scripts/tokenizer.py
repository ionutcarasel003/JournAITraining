from transformers import AutoTokenizer

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )
    
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer