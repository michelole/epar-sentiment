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
MIN_DF = 0.01
MAX_DF = 0.99

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=MIN_DF, max_df=MAX_DF)
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
# ## Text classification
# %% [md]
# ### Using SVM + tf-idf
# %% [md]
# Declare a pipeline.
# %%
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC

svm_clf = Pipeline([
     ('vect', CountVectorizer(min_df=MIN_DF, max_df=MAX_DF)),
     ('tfidf', TfidfTransformer()),
     # XXX Probability may slow down model
     ('clf', SVC(kernel='linear', probability=True))
 ])

# %% [md]
# Cross-validate the dataset. This generates more robust metrics and allow us to use the full
# dataset for both training and evaluation.
# %%
from sklearn.model_selection import cross_val_score
from statistics import mean, stdev

scores = cross_val_score(svm_clf, data['clean_sentence'], data['rating'], cv=10)
print(mean(scores))
print(stdev(scores))

# %% [md]
# Generate confusion matrix.
# %%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

y_pred = cross_val_predict(svm_clf, data['clean_sentence'], data['rating'], cv=10)
conf_matrix = confusion_matrix(data['rating'], y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='OrRd')
plt.title(f"SVM, min_df = {MIN_DF}, max_df = {MAX_DF}")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

# %% [md]
# Display classification report.
# %%
from sklearn.metrics import classification_report

print(classification_report(data['rating'], y_pred))

# %% [md]
# ### Using `fastText`

# %% [md]
# Train and fine-tune the classifier.
# %%
from skift import ColLblBasedFtClassifier

DIM = [1, 2, 5, 10, 20, 50, 100]
EPOCHS = [1, 2, 5, 10, 20, 50, 100]
LR = [0.01, 0.1, 0.2, 0.5, 1.0]

max_score = 0
stdev_max_score = 0
dim_max_score = 0
epochs_max_score = 0
lr_max_score = 0

# Manually run grid search since `skift` does not support sklearn's `GridSearchCV`
for dim in DIM:
    for epoch in EPOCHS:
        for lr in LR:
            print(f"dim={dim}, epoch={epoch}, lr={lr}")
            ft_clf = ColLblBasedFtClassifier(input_col_lbl='clean_sentence', dim=dim, epoch=epoch, lr=lr)
            scores = cross_val_score(ft_clf, data[['clean_sentence']], data['rating'], cv=10)
            mean_score = mean(scores)
            stdev_score = stdev(scores)
            if mean_score > max_score:
                print(f"{mean_score} +- {stdev_score}")
                print(stdev_score)
                max_score = mean_score
                stdev_max_score = stdev_score
                dim_max_score = dim
                epochs_max_score = epoch
                lr_max_score = lr

print("Best model:")
print(f"dim={dim_max_score}, epoch={epochs_max_score}, lr={lr_max_score}")
print(f"{max_score} +- {stdev_max_score}")

# %%
