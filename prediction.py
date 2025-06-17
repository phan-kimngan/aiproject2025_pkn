from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import json
import torch
import re
from pathlib import Path
from konlpy.tag import Okt
import joblib 
    
# You can try the available model here
#model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
#tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('fine_tuned_digit82_kobart-2')
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')



def preprocess_korean(texts):    
    # Convert to lowercase (Optional for Korean)
    text = texts.lower()        
    # Replace multiple spaces with a single space
    #text = re.sub(r'\s+', ' ', text).strip()
    # Remove repeated/Special characters, keep only the Korean characters, numbers, and space
    text = re.sub(r'[^가-힣0-9\s]', '', text)     

    return text


def okt_preprocess(texts):
    # Tokenization using Okt
    pattern = r'\b[가-힣]{2,5}[:,]\s*'
    text = re.sub(pattern, '', texts)
    okt = Okt()
    tokens = okt.morphs(text)    
    # Define a simple Korean stopword list
    stopwords = [
    # Pronouns
    "나", "너", "저", "우리", "그", "그녀",
    
    # Particles
    "은/는", "이/가", "을/를", "에", "에서", "와/과", "하고", "의",
    
    # Verbs
    "하다", "있다", "없다", "가다", "오다", "보다", "말하다", "듣다",
    
    # Nouns
    #"사람", "시간", "것", "이야기", "친구", "사랑"
    ]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords]
    tokens = ' '.join(tokens)
    
    return tokens
def get_summary(text, max_input_length=1024, max_output_length=100):
    text = preprocess_korean(text)
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=max_input_length, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_output_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
text = '요즘 왜 그렇게 생각이 많아 보여? 그냥… 연애가 너무 어렵다는 생각이 들어서. 무슨 일 있었어? 요즘 남자친구랑 자꾸 싸워. 사소한 일에도 서로 예민하게 반응하고, 대화가 점점 줄어드는 느낌이야. 그런 시기 누구나 오는 것 같아. 근데 감정 쌓아두지 말고 솔직하게 이야기해보는 게 중요해. 나도 그게 맞는 줄은 아는데, 자꾸 말을 꺼내는 게 무서워. 혹시 상처 줄까 봐. 네가 걱정하는 만큼 상대도 너를 걱정하고 있을 수도 있어. 한 번은 용기 내보는 것도 필요하지 않을까? 맞아… 고마워, 지은아. 너랑 얘기하니까 마음이 좀 가벼워졌어.'
get_pre_summary = get_summary(text)
print('get_pre_summary', get_pre_summary)
    
