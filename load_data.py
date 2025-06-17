import json
from pprint import pprint
import numpy as np
import torch
import re
from konlpy.tag import Okt
from gensim.models import Word2Vec
import torch
from konlpy.tag import Okt
from datasets import Dataset
import sentencepiece as spm
from sklearn.model_selection import train_test_split
#from konlpy.tag import Mecab
import sentencepiece as spm
with open('./Training/[라벨]한국어대화요약_train/[라벨]한국어대화요약_train/개인및관계.json', 'r', encoding='utf-8') as files:
    train_json_data= json.load(files)
    files.close()

train_utterances = []
for dialogue in train_json_data["data"]:
    train_utterances.append([utterance["utterance"]  for utterance in dialogue["body"]["dialogue"]])
train_summary = [dialogue["body"]["summary"] for dialogue in train_json_data["data"]]

with open('./Validation/[라벨]한국어대화요약_valid/[라벨]한국어대화요약_valid/개인및관계.json', 'r', encoding='utf-8') as files:
    val_json_data= json.load(files)
    files.close()
test_utterances = []
for dialogue in val_json_data["data"]:
    test_utterances.append([utterance["utterance"]  for utterance in dialogue["body"]["dialogue"]])
test_summary = [dialogue["body"]["summary"] for dialogue in val_json_data["data"]]


def preprocess_korean(texts):
    text = texts.lower()        
    text = re.sub(r'[^가-힣0-9\s]', '', text)     

    return text

    
train_preprocessed_utterances = []
i = 0
for utt in train_utterances:
    train_preprocessed_utterances.append([preprocess_korean(u) for u in utt])
summaries = [preprocess_korean(s) for s in train_summary]

inputs = []
for sentence in train_preprocessed_utterances:
    inputs.append(' '.join(sentence))
    

data_train, data_val, sum_train, sum_val = train_test_split(inputs, summaries, test_size=0.2, random_state=42)
with open('./data/data_train2.txt', 'w') as f:
    f.write(json.dumps(data_train))
with open('./data/data_val2.txt', 'w') as f:
    f.write(json.dumps(data_val))
with open('./data/sum_train2.txt', 'w') as f:
    f.write(json.dumps(sum_train))
with open('./data/sum_val2.txt', 'w') as f:
    f.write(json.dumps(sum_val))
    
    
    

