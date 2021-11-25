import pandas as pd
import numpy as np
import csv
from nltk.tokenize import word_tokenize 


data = pd.read_csv("refined_alldata.csv", nrows=50, usecols = ['full_text'])
defined_data = data.full_text.tolist()

from rank_bm25 import BM25Okapi
tokenized_corpus = [doc.split(" ") for doc in defined_data]
bm25 = BM25Okapi(tokenized_corpus)
query = "Corona virus pandemic"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)
bm25.get_top_n(tokenized_query, defined_data, n=3)