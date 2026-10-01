from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model_and_tokenizer(model_name, num_labels, label2id, id2label):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    model.config.problem_type = "multi_label_classification"
    return tokenizer, model
