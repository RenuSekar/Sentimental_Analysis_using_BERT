from transformers import AutoModelForSequenceClassification,AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("./saved_model")
tokenizer = AutoTokenizer.from_pretrained("./saved_model")

def sentiment_score(sentence):
    tokens = tokenizer.encode(sentence,return_tensors = 'pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1