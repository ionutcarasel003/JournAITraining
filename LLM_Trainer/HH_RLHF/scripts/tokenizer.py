from transformers import AutoTokenizer

def load_tokenizer(model_name):
    """
    Loads and configures the tokenizer for the base model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Set padding side to right for training stability
    tokenizer.padding_side = "right"

    return tokenizer
