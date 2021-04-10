# epar-sentiment: Sentiment analysis of EPARs

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Context
Pharmaceutical companies submit drug applications to the European Medicines Agency (EMA) containing scientific evidence and clinical trial data. The EMA then assesses the applications and publishes European Public Assessment Reports (EPARs), PDF documents containing the supporting evidence for approval or refusal.

Analyzing these documents (preferably automatically) may reveal patterns that increases the chance of approval. A first step is then to identify the sentences containing positive or negative sentiments.

## Goal

Evaluate the feasibility of automatic sentiment analysis on a subset of randomly collected sentences.

## Development

- Add the `sentences_with_sentiment.xlsx` file to the data folder
- Install Python 3.7.10 using a virtual environment
- `pip install -r requirements.txt`
- `pip install -r dev-requirements.txt`