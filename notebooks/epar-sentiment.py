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
# Select features based on document frequency.

# %%
MIN_DF = 0.01
MAX_DF = 0.99

# %% [md]
# Handcrafted features.

CANDIDATE_NEGATIVE_FEATURES = set(["missing", "further", "awaited", "address", "issues", "limited",
    "questions", "weak", "inconsistent", "poor", "requested", "uncertainties", "additional", "lack",
    "questionable"])
CANDIDATE_NEUTRAL_FEATURES = set(["conditional", "final", "additional", "long", "term", "common",
    "events"])
CANDIDATE_POSITIVE_FEATURES = set(["meaningful", "significant", "response", "activity", "support",
    "efficacy", "consistent", "biosimilarity", "bioequivalence", "bioequivalent", "demonstrate",
    "demonstrated", "comparable", "comparability", "compared", "acceptable", "accepted",
    "supportive", "superiority", "sufficient", "extrapolated", "extrapolation", "similar",
    "similarity", "similarly", "adequate", "line", "statistically", "appropriate", "safety",
    "relevant", "favourable", "approvable"])
FEATURES = CANDIDATE_NEGATIVE_FEATURES.union(CANDIDATE_NEUTRAL_FEATURES).union(CANDIDATE_POSITIVE_FEATURES)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# XXX Switch from frequency-based to handcrafted features.
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=MIN_DF, max_df=MAX_DF)
# vectorizer = TfidfVectorizer(sublinear_tf=True, vocabulary=FEATURES)
# vectorizer = TfidfVectorizer(sublinear_tf=True)

X_train = vectorizer.fit_transform(data['clean_sentence'])
print(X_train.shape)

feature_names = vectorizer.get_feature_names()
print(feature_names)

document_matrix = pd.DataFrame.sparse.from_spmatrix(X_train).sparse.to_dense()
document_matrix.columns = feature_names

train_features = pd.concat([data.reset_index(), document_matrix], axis=1).sort_values('rating')\
    .drop(columns=['index', 'Sentence', 'Positive', 'Negative', 'Neutral', 'sentence_length'])

train_features.to_excel('../features.xlsx')

# %% [md]
# ## Feature selection
# %% [md]
# Select features using RFE (recursive feature elimination).
# %%
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

TARGET_FEATURES = 100

# Column 3 is where the term-incidence matrix starts
input_features = train_features.iloc[:, 3:]
svc = SVC(kernel='linear')
selector = RFE(svc, n_features_to_select=TARGET_FEATURES)
selector = selector.fit(input_features, train_features['rating'])

candidate_vocabulary = np.array(feature_names, dtype=object)
selected_vocabulary = candidate_vocabulary[selector.get_support()]
print(selected_vocabulary)
print(len(selected_vocabulary))

# Prefer to use CV next
# selector.score(input_features, train_features['rating'])
    
# %% [md]
# ## Text classification

# %% [md]
# Since our dataset is imbalanced, calculate a majority baseline.
# %%
from sklearn.dummy import DummyClassifier

majority_clf = DummyClassifier(random_state=42)

# %% [md]
# Using SVM + tf-idf
# %%
from sklearn.pipeline import Pipeline

svm_clf = Pipeline([
     ('tfidf', TfidfVectorizer(vocabulary=selected_vocabulary, sublinear_tf=True)),
     # XXX Probability may slow down model
     ('clf', SVC(kernel='linear', probability=True))
 ])

# %% [md]
# Train and fine-tune a `fastText` classifier.
# %%
from skift import ColLblBasedFtClassifier
from sklearn.model_selection import cross_val_score
from statistics import mean, stdev

DIM = [20]
EPOCHS = [100]
LR = [1.0]
# XXX Uncomment to run grid search
# DIM = [1, 2, 5, 10, 20, 50, 100]
# EPOCHS = [1, 2, 5, 10, 20, 50, 100]
# LR = [0.01, 0.1, 0.2, 0.5, 1.0]

mean_max_score = 0
max_ft_scores = []
dim_max_score = 0
epochs_max_score = 0
lr_max_score = 0
best_ft_clf = False

# Manually run grid search since `skift` does not support sklearn's `GridSearchCV`
for dim in DIM:
    for epoch in EPOCHS:
        for lr in LR:
            print(f"dim={dim}, epoch={epoch}, lr={lr}")
            ft_clf = ColLblBasedFtClassifier(input_col_lbl='clean_sentence', dim=dim, epoch=epoch, lr=lr)
            ft_scores = cross_val_score(ft_clf, data[['clean_sentence']], data['rating'], cv=10)
            mean_score = mean(ft_scores)
            stdev_score = stdev(ft_scores)
            if mean_score > mean_max_score:
                print(f"{mean_score} +- {stdev_score}")
                best_ft_clf = ft_clf
                mean_max_score = mean_score
                max_ft_scores = ft_scores
                dim_max_score = dim
                epochs_max_score = epoch
                lr_max_score = lr

print(f"Best fastText model: dim={dim_max_score}, epoch={epochs_max_score}, lr={lr_max_score}")

# %% [md]
# Iterate over the declared classifiers to generate reports.
# %%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from util import plot_confusion_matrix
from util import explain

EXPLAIN_ID = 4
SCORING = 'f1_micro'

classifiers = {'Majority': majority_clf, 'SVM': svm_clf, 'fastText': best_ft_clf}
scores = {'Majority': [], 'SVM': [], 'fastText': []}

for k, v in classifiers.items():
    X_train = data['clean_sentence']
    instance = X_train[EXPLAIN_ID - 1]
    method = v.predict_proba

    # `skift` is really a bad wrapper library...
    if k == 'fastText':
        X_train = data[['clean_sentence']]
        method = v.predict_proba_on_str_arr
    
    # Cross-validate the dataset.
    clf_scores = cross_val_score(v, X_train, data['rating'], scoring=SCORING, cv=10)
    scores[k] = clf_scores
    print(f"{k}: {mean(clf_scores)} +- {stdev(clf_scores)}")

    # Generate confusion matrix.
    pred = cross_val_predict(v, X_train, data['rating'], cv=10)
    plot_confusion_matrix(data['rating'], pred, k)

    # Display classification report.
    print(classification_report(data['rating'], pred))

    # Use LIME to explain predictions.
    explain(v, X_train, data['rating'], instance, k, method)

# %% [md]
# ### Overall results

# %% [md]
# Compare overall classification results using boxplot.
# %%
from matplotlib import pyplot as plt

# TODO generate automatically based on `scores`
results = pd.concat([pd.DataFrame({SCORING: scores['Majority'], 'method': "Majority"}),
    pd.DataFrame({SCORING: scores['fastText'], 'method': "fastText"}),
    pd.DataFrame({SCORING: scores['SVM'], 'method': "SVM"})])

results.boxplot(column=[SCORING], by='method', positions=[1, 3, 2], showmeans=True, figsize=(7, 5))
plt.ylim([0.4, 1.0])

# %% [md]
# ### Using Linear Regression

# %% [md]
# Train a Ridge regressor.

# %%
from sklearn.linear_model import Ridge
ridge_regr = Pipeline([
     ('tfidf', TfidfVectorizer()),
     ('regr', Ridge())
 ])

for scoring in ['r2', 'neg_root_mean_squared_error']:
    svm_scores = cross_val_score(ridge_regr, data['clean_sentence'], data['rating'], cv=10, scoring=scoring)
    print(scoring)
    print(mean(svm_scores))
    print(stdev(svm_scores))

# %%
