# Natural Language Processing (NLP) with Python (Spamd Detection)

This repository contains a Python-based implementation of Natural Language Processing (NLP) techniques to classify SMS messages as either "ham" (legitimate messages) or "spam" (unwanted messages). The project uses the **NLTK** library for text processing and **Scikit-Learn** for machine learning.

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Text Preprocessing](#text-preprocessing)
6. [Vectorization](#vectorization)
7. [Model Training](#model-training)
8. [Model Evaluation](#model-evaluation)
9. [Pipeline Creation](#pipeline-creation)
10. [Resources](#resources)

## Introduction
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. In this project, we use NLP techniques to preprocess text data and train a machine learning model to classify SMS messages as either "ham" or "spam".

## Requirements
To run this project, you need the following Python libraries installed:
- **NLTK**: For text processing and tokenization.
- **Pandas**: For data manipulation and analysis.
- **Scikit-Learn**: For machine learning and model evaluation.
- **Matplotlib** and **Seaborn**: For data visualization.

You can install the required libraries using the following command:
```bash
pip install nltk pandas scikit-learn matplotlib seaborn
```

Additionally, you need to download the NLTK stopwords corpus:
```python
import nltk
nltk.download('stopwords')
```

## Dataset
The dataset used in this project is from the **UCI Machine Learning Repository**. It contains over 5,000 SMS messages, each labeled as either "ham" or "spam". The dataset is stored in a tab-separated values (TSV) file.

## Exploratory Data Analysis (EDA)
We perform basic exploratory data analysis to understand the dataset:
- **Message Length Analysis**: We analyze the length of messages to see if there is a difference in length between "ham" and "spam" messages.
- **Label Distribution**: We check the distribution of "ham" and "spam" messages in the dataset.

## Text Preprocessing
Before feeding the text data into a machine learning model, we preprocess it by:
1. **Removing Punctuation**: We remove all punctuation from the text.
2. **Tokenization**: We split the text into individual words (tokens).
3. **Removing Stopwords**: We remove common words (e.g., "the", "a", "and") that do not contribute much to the meaning of the text.

## Vectorization
To convert the text data into a format that machine learning models can understand, we use the **Bag-of-Words** model:
1. **CountVectorizer**: Converts the text into a matrix of token counts.
2. **TF-IDF Transformer**: Applies Term Frequency-Inverse Document Frequency (TF-IDF) weighting to the token counts.

## Model Training
We use the **Naive Bayes** classifier, which is a popular choice for text classification tasks. The model is trained on the preprocessed and vectorized text data.

## Model Evaluation
We evaluate the model's performance using metrics such as **precision**, **recall**, and **F1-score**. The model is tested on a separate test set to ensure that it generalizes well to unseen data.

## Pipeline Creation
To streamline the process of preprocessing and model training, we create a **data pipeline** using Scikit-Learn's `Pipeline` class. This allows us to chain together the preprocessing steps and the classifier into a single object.

## Resources
For more information on Natural Language Processing and the tools used in this project, check out the following resources:
- [NLTK Book Online](https://www.nltk.org/book/)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Kaggle NLP Walkthroughs](https://www.kaggle.com/learn/natural-language-processing)

---

This project provides a basic introduction to NLP and text classification using Python. Feel free to explore the code and experiment with different models and techniques!