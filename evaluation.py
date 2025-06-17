from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import json
import torch
import re
from pathlib import Path
 
    
# You can try the available model here
#model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
#tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('fine_tuned_digit82_kobart-2')
tokenizer = PreTrainedTokenizerFast.from_pretrained('fine_tuned_digit82_kobart-2')




def compute_bleu2(reference, prediction):
    sf = SmoothingFunction(epsilon=1e-12).method1
    ref_tokens = [word_tokenize(reference)]
    pred_tokens = word_tokenize(prediction)
    b1 = sentence_bleu(references, hypothesis, weights=(1.0/1.0,), smoothing_function=sf)
    b2 = sentence_bleu(references, hypothesis, weights=(1.0/2.0, 1.0/2.0), smoothing_function=sf)
    b3 = sentence_bleu(references, hypothesis, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0), smoothing_function=sf)
    b4 = sentence_bleu(references, hypothesis, weights=(1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0), smoothing_function=sf)
    
    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
    return score

def compute_bleu(reference, prediction):
    smoothie = SmoothingFunction().method4
    ref_tokens = [word_tokenize(reference)]
    pred_tokens = word_tokenize(prediction)
    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
    return score

def get_summary(text, max_input_length=1024, max_output_length=100):
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=max_input_length, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_output_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

with open('./Validation/[라벨]한국어대화요약_valid/[라벨]한국어대화요약_valid/개인및관계.json', 'r', encoding='utf-8') as files:
    val_json_data= json.load(files)
    files.close()
test_utterances = []
for dialogue in val_json_data["data"]:
    test_utterances.append([utterance["utterance"]  for utterance in dialogue["body"]["dialogue"]])
test_summary = [dialogue["body"]["summary"] for dialogue in val_json_data["data"]]

def preprocess_korean(texts):
    
    # Convert to lowercase (Optional for Korean)
    text = texts.lower()        
    # Replace multiple spaces with a single space
    #text = re.sub(r'\s+', ' ', text).strip()
    # Remove repeated/Special characters, keep only the Korean characters, numbers, and space
    text = re.sub(r'[^가-힣0-9\s]', '', text)     

    return text

test_utterances = [preprocess_korean(' '.join(dialogue)) for dialogue in test_utterances]
test_summary = [preprocess_korean(dialogue) for dialogue in test_summary]


predicted_summary = []
bleu_score = []
for i in range(len(test_utterances)):
     print('Number', i)
     get_pre_summary = get_summary(test_utterances[i])
     predicted_summary.append(get_pre_summary)
     bleu = compute_bleu(test_summary[i], get_pre_summary)
     bleu_score.append(bleu)
     print('Predicted summary', get_pre_summary)
     print('Reference summary', test_summary[i])
     print('bleu score', bleu)
     print("\n")

tensor_bleu_score = torch.tensor(bleu_score, dtype=torch.float32)
bleu_score = tensor_bleu_score.mean()
print('bleu_score', predicted_summary, bleu_score)




outdir = Path("./results")
outdir.mkdir(exist_ok=True, parents=True)

output_prediction_file =  outdir / "predictions_acc-2.txt"
         
with output_prediction_file.open('w') as f:  
     f.write("Bleu Score" + str(bleu_score) + "\n")      
     for i, l1 in enumerate(predicted_summary):
          f.write("Pred" + str(predicted_summary[i]) + "\n")            
          f.write("Gold" + str(test_summary[i]) + "\n")
    
