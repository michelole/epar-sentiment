# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [md]
# # Sentiment analysis of EPARs
# %% [md]
# Load the data.
# %%
import pandas as pd
from pandas.core.arrays import base

data = pd.read_excel("../data/sentences_with_sentiment.xlsx")
print(data.dtypes)
data.head()

# %% [md]
# Ensure that we have a single rating per sentence.
# %%
assert max(data['Positive'] + data['Negative'] + data['Neutral']) == 1
assert min(data['Positive'] + data['Negative'] + data['Neutral']) == 1

# %% [md]
# ## Descriptive analytics
# %% [md]
# Count ratings.
# %%
positive = sum(data['Positive'])
negative = sum(data['Negative'])
neutral = sum(data['Neutral'])
print(positive)
print(negative)
print(neutral)

# %% [md]
# Check length distribution.
# %%
lengths = data['Sentence'].str.len()
print(lengths.describe())
lengths.plot.box()

# %% [md]
# Set a majority baseline.
# %%
baseline_acc = max(positive, negative, neutral) / len(data)
print(baseline_acc)

# %% [md]
# ## Pre-processing
# %% [md]
# Lowercase and keep only alphabetic characters (including e.g. the German umlaut).
# Numbers do not seem to convey meaning and commonly lead to overfitting.
# %%
import re
# Lowercase
data['clean_sentence'] = data['Sentence'].str.lower()

# \W = [^a-zA-Z0-9_] + Unicode variations
cleaner = re.compile(r"[0-9_\W\s]+")
data['clean_sentence'] = data['clean_sentence'].str.replace(cleaner, " ")

data['clean_sentence']

# %% [md]
# Remove stopwords.
# %%
import nltk

# TODO Check the impact of removing "should" (and its variations), since it seems to be a predictor
# for negative/neutral ratings.
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

def remove_stopwords(text):
    tokens = text.split()
    return " ".join([word for word in tokens if word not in stopwords])

data['clean_sentence'] = data['clean_sentence'].apply(remove_stopwords)

data['clean_sentence']
