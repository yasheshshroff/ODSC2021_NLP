#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

cwd = Path(__file__).parent
data_path = cwd / 'disaster_data/predict.csv'
vec_path = cwd / 'vectorizer.pk'
clf_path = cwd / 'clf.pk'

# queries are stored in the variable query_text
query_text = pd.read_csv(data_path).text.values

with open(vec_path, 'rb') as fin:
    vectorizer= pickle.load(fin)

with open(clf_path, 'rb') as fin:
    clf = pickle.load(fin)


def predict(model, query_txt):
    x = vectorizer.transform([query_txt]).toarray()
    pred = model.predict(x)
    if pred[0] == 1:
        print(f"[red]{query_txt} -> Disaster")
    else:
        print (f"[green]{query_txt} -> Not a disaster")


for query in query_text:
    predict (clf, query)

