import json
from pprint import pprint
import numpy as np
import torch
import re
from konlpy.tag import Okt
from gensim.models import Word2Vec
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, GPT2LMHeadModel
from konlpy.tag import Okt
from datasets import Dataset
import sentencepiece as spm
from transformers import PreTrainedTokenizerFast
from transformers import AdamW
from sklearn.model_selection import train_test_split
#from konlpy.tag import Mecab
import sentencepiece as spm


with open('./data/data_train.txt', 'r') as f:
    data_train = json.loads(f.read())
with open('./data/data_val.txt', 'r') as f:
    data_val = json.loads(f.read())
with open('./data/sum_train.txt', 'r') as f:
    sum_train = json.loads(f.read())
with open('./data/sum_val.txt', 'r') as f:
    sum_val = json.loads(f.read())

train_data= {}
train_data["text"] = data_train
train_data["summary"] = sum_train
train_dataset = Dataset.from_dict(train_data)


val_data= {}
val_data["text"] = data_val
val_data["summary"] = sum_val
val_dataset = Dataset.from_dict(val_data)

print(len(data_train), len(train_data["text"] ))

# Define the preprocessing function
def preprocess_function(examples):
    inputs = [doc for doc in examples["text"]]  # Tokenize the document (input)
    targets = [summary for summary in examples["summary"]]  # Tokenize the summary (output)
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# You can try the available model here
# model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
# tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('fine_tuned_digit82_kobart-2')
tokenizer = PreTrainedTokenizerFast.from_pretrained('fine_tuned_digit82_kobart-2')

# Preprocess the dataset
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,
                  lr=2e-5, correct_bias=False)
#222222222222222222222222222222222222222
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",        # Evaluate at the end of every epoch
    save_strategy="epoch",              # Save checkpoint every epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    save_total_limit=1,                 # Keep only the best model
    # Important for summarization
    fp16=True,                          # Enable mixed precision if using GPU
    logging_dir="./logs",
    load_best_model_at_end=True,
    greater_is_better=True,
)
#optimizer = AdamW(model.parameters(), lr=2e-5)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, None)
)
# Fine-tune the model
trainer.train()
model.save_pretrained("fine_tuned_digit82_kobart-2")
tokenizer.save_pretrained("fine_tuned_digit82_kobart-2")
