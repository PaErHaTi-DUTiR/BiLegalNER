import os
import numpy as np
import transformers
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import AutoTokenizer, XLMRobertaForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset, load_metric
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from seqeval.scheme import IOB2
import seqeval.metrics

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

task = "ner"
datasets = load_dataset("./ugnerd.py")
label_list = datasets["train"].features[f"{task}_tags"].feature.names

model_checkpoint = "cino-large-v2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

model = XLMRobertaForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
# 模型名	        MODEL_NAME
# CINO-large-v2	hfl/cino-large-v2
# CINO-base-v2	hfl/cino-base-v2
# CINO-small-v2	hfl/cino-small-v2
# CINO-large	hfl/cino-large

assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast) 

data_collator = DataCollatorForTokenClassification(tokenizer)

def tokenize_and_align_labels(examples, tokenizer, task, label_all_tokens=False):
    max_length = 128
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        padding=True,
        max_length=max_length,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels, scheme="IOB2", mode="strict")
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

tokenized_datasets = datasets.map(
    lambda examples: tokenize_and_align_labels(examples, tokenizer=tokenizer, task=task),
    batched=True
)


training_args = TrainingArguments(
    output_dir="./{model_checkpoint}-finetuned-ner",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=16,
    learning_rate=2e-4,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator
)


trainer.train()

trainer.evaluate()

print("#" * 100)
predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
predictions = np.argmax(predictions, axis=2)

true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)]
true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
               for prediction, label in zip(predictions, labels)]

results = metric.compute(predictions=true_predictions, references=true_labels, scheme="IOB2", mode="strict")
print("\nValidation: Overall F1", results["overall_f1"])
print("Validation: Total number of sentences:", len(true_labels))
print("Validation: Total number of tokens:", sum([len(sent) for sent in true_labels]))
print("Validation: seqeval based results")
print(seqeval.metrics.classification_report(true_labels, true_predictions, digits=4, mode='strict', scheme=IOB2))


print("#" * 100)
predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=2)

true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)]
true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
               for prediction, label in zip(predictions, labels)]

results = metric.compute(predictions=true_predictions, references=true_labels, scheme="IOB2", mode="strict")
print("\nTest: Overall F1", results["overall_f1"])
print("Test: Total number of sentences:", len(true_labels))
print("Test: Total number of tokens:", sum([len(sent) for sent in true_labels]))
print("Test: seqeval based results")
print(seqeval.metrics.classification_report(true_labels, true_predictions, digits=4, mode='strict', scheme=IOB2))
