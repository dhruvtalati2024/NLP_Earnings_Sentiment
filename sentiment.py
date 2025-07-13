from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def run_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def run_finbert(text):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    CHUNK_SIZE = 450
    sentences = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    all_probs = []
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
        all_probs.append(probs)
    avg_probs = sum([torch.tensor(x) for x in all_probs]) / len(all_probs)
    labels = ['negative', 'neutral', 'positive']
    sent = {label: float(avg_probs[i]) for i, label in enumerate(labels)}
    sent['pos_minus_neg'] = sent['positive'] - sent['negative']
    return sent

def run_loughran_mcdonald(text):
    positive = set([
        'growth','increase','record','improve','profit','gain','benefit','strong','achieve','success','expand','opportunity','innovation'
    ])
    negative = set([
        'risk','uncertain','decrease','loss','cost','decline','negative','drop','difficult','challenge','weak','pressure','impact','tariff'
    ])
    words = text.split()
    pos_count = sum(1 for w in words if w in positive)
    neg_count = sum(1 for w in words if w in negative)
    total = len(words)
    return {
        'lm_positive_pct': pos_count / total * 100,
        'lm_negative_pct': neg_count / total * 100,
        'lm_score': (pos_count - neg_count) / total * 100
    }
