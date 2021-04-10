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
# Normalize ratings in a single column.

# %%
data['rating'] = 1*data['Positive'] + 0*data['Neutral'] - 1*data['Negative']

# %% [md]
# Check length distribution.
# %%
data['sentence_length'] = data['Sentence'].str.len()
print(data['sentence_length'].describe())
data['sentence_length'].plot.box()

# %% [md]
# Does length vary by rating?
# %%
data.boxplot(column=['sentence_length'], by='rating', figsize=(5, 7))

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

# %% [md]
# ## Textual analysis
# %% [md]
# How does the vocabulary affect the rating?
# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=0.01, max_df=0.99)
X_train = vectorizer.fit_transform(data['clean_sentence'])
print(X_train.shape)

feature_names = vectorizer.get_feature_names()
print(feature_names)

document_matrix = pd.DataFrame.sparse.from_spmatrix(X_train).sparse.to_dense()
document_matrix.columns = feature_names

train_features = pd.concat([data.reset_index(), document_matrix], axis=1).sort_values('rating')\
    .drop(columns=['index', 'Sentence', 'Positive', 'Negative', 'Neutral', 'sentence_length'])

train_features.to_excel('../features.xlsx')
