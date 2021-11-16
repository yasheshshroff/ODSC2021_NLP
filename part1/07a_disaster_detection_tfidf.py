#!/usr/bin/env python
# coding: utf-8

# ## Disaster or not: Text Classification using TFIDF and Logistic Regression

import numpy as np
import pandas as pd
import pickle


# ### Load data

from pathlib import Path

cwd = Path(__file__).parent
data_path = cwd / "disaster_data/train.csv"
pd.read_csv(data_path).head()


# queries are stored in the variable query_text
# correct intent labels are stored in the variable labels

query_text = pd.read_csv(data_path).text.values
labels = pd.read_csv(data_path).target.values

query_text.shape


# ### Train and Test split

from sklearn.model_selection import train_test_split

query_train, query_test, y_train, y_test = train_test_split(query_text, labels, test_size=0.2, random_state=13)


# ### Vectorize the text document

from sklearn.feature_extraction.text import TfidfVectorizer

ngram_range = (1,2)

vectorizer = TfidfVectorizer(ngram_range=ngram_range, 
                             stop_words='english', 
                             max_features=150)

X_train = vectorizer.fit_transform(query_train).toarray()
X_test = vectorizer.transform(query_test).toarray()

# ### Fit a classifier using the vectors

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)

vec_path = cwd / 'vectorizer.pk'
clf_path = cwd / 'clf.pk'

with open(vec_path, 'wb') as fin: 
    pickle.dump(vectorizer, fin)

with open(clf_path, 'wb') as fin: 
    pickle.dump(clf, fin)




