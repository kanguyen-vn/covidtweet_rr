import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi

# all the tweets from the all_tweetdata.csv file
data = pd.read_csv("../data/all_tweetdata.csv", usecols = ['processed'])
defined_data = data.processed

#we filter out the tweets that are empty
defined_data = [item for item in defined_data if str(item) != 'nan']

tokenized_corpus = [doc.split(" ") for doc in defined_data]

#used the BM25Okapi library by PyPI
bm25 = BM25Okapi(tokenized_corpus)

query_data = pd.read_csv("../data/all_querydata.csv", usecols = ['title'])
defined_querydata = query_data.title
defined_querydata = [item for item in defined_querydata if str(item) != 'nan']

result_list = []

for q in defined_querydata:
    tokenized_query = q.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    result_list.append(doc_scores)

print(result_list)