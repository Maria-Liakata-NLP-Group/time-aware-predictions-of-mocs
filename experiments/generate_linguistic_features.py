import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer


import sys
sys.path.insert(0, "../../timeline_generation/")  # Adds higher directory to python modules path

from utils.io.my_pickler import my_pickler


dataset='talklife'

df = my_pickler("i", "{}_timelines_all_embeddings".format(dataset), folder='datasets')

print("Running sentiment analysis...")
sia = SentimentIntensityAnalyzer()

# Assign sentiment scorees, using VADER
df['sentiment_vader'] = df['content'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Assign labels based on vader sentiment scores
df['sentiment_label'] = np.nan
df.loc[df['sentiment_vader'] > 0.00000, 'sentiment_label'] = 'positive'
df.loc[df['sentiment_vader'] < 0.00000, 'sentiment_label'] = 'negative'

"""
==== EMOTION ====
"""
"""
Download and set up initial requirements for the emotion classifier.
"""
task='emotion'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# download label mapping
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

    
def return_emotions_from_text(text, verbose=False, return_only_maximum_emotion=False):
    """
    Takes an input string, and outputs the emotion scores.
    """
    
    # Can only process text data.
    if type(text) != str:
        return text
    try:
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        output = {}
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            output[l] = s

            if verbose:
                print(f"{i+1}) {l} {np.round(float(s), 4)}")

        output = pd.Series(output)

        if return_only_maximum_emotion:
            return output.idxmax()
        else:
            return output
    except:
#         return np.NaN
        return pd.Series(data=[np.NaN, np.NaN, np.NaN, np.NaN], index=['anger', 'sadness', 'optimism', 'joy'])
    
    
print('Running emotion classifier...')
moods = df['content'].apply(lambda x: return_emotions_from_text(x))

df_linguistic_features_moods_sentiment = pd.concat((df, moods), axis=1)

# Assign the mode mood score as the mood
df_linguistic_features_moods_sentiment['mood'] = df_linguistic_features_moods_sentiment[['joy', 'sadness', 'optimism', 'anger']].idxmax(axis=1)

my_pickler("o", "df_{}_linguistic".format(dataset), df_linguistic_features_moods_sentiment, folder='datasets')