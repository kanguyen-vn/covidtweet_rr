import os
import sys
from shutil import copytree, rmtree
import logging
import requests
import zipfile
import pandas as pd
from ast import literal_eval
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np


logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_tweets_dir = os.path.join(current_dir, "processed_tweets")
tfidf_dir = os.path.join(current_dir, "tf_idf")
try:
    os.mkdir(tfidf_dir)
except FileExistsError:
    pass

embeddings_dir = os.path.join(current_dir, "embeddings")
try:
    os.mkdir(embeddings_dir)
except FileExistsError:
    pass

glove_dir = os.path.join(os.path.dirname(current_dir), "glove")
try:
    os.mkdir(glove_dir)
except FileExistsError:
    pass

model_input_dir = os.path.join(current_dir, "model_input")
try:
    os.mkdir(model_input_dir)
except FileExistsError:
    pass

claim_files = ["ClaimFakeCOVID-19.csv", "ClaimRealCOVID-19.csv"]
news_files = ["NewsFakeCOVID-19.csv", "NewsRealCOVID-19.csv"]
tweet_files = [f"{pref}COVID-19_tweets_expanded.csv" for pref in [
    "ClaimFake", "ClaimReal", "NewsFake", "NewsReal"]]
seed = 27


def copy(input_dir, output_dir):
    logger.info(f"Copying files from {input_dir} to {output_dir}...")
    if os.path.exists(output_dir):
        rmtree(output_dir)
    copytree(input_dir, output_dir, dirs_exist_ok=True)


def concatenate_news(input_dir, output_dir):
    logger.info("Concatenating different fields in news...")
    for filename in news_files:
        logger.info(f"- Processing {filename}...")
        filepath = os.path.join(input_dir, filename)
        output_filepath = os.path.join(output_dir, filename)
        df = pd.read_csv(filepath, header=0)
        concatenated = []
        for i in range(len(df)):
            concat_doc = " ".join(
                [df.iat[i, j] for j in range(1, 5) if not pd.isnull(df.iat[i, j])])
            concatenated.append(concat_doc)
        df["concatenated"] = concatenated
        df.to_csv(output_filepath, index=False)


def delete_empty_queries_all(input_dir, output_dir):
    delete_empty_queries("title", claim_files, input_dir, output_dir)
    delete_empty_queries("concatenated", news_files, input_dir, output_dir)


def delete_empty_queries(column, query_files, input_dir, output_dir):
    for filename in query_files:
        logger.info(f"Deleting empty queries in {filename}...")
        filepath = os.path.join(input_dir, filename)
        output_filepath = os.path.join(output_dir, filename)
        tweet_filepath = os.path.join(
            input_dir, f"{filename[:-4]}_tweets_expanded.csv")
        output_tweet_filepath = os.path.join(
            output_dir, f"{filename[:-4]}_tweets_expanded.csv")

        query_df = pd.read_csv(filepath, header=0)
        empty_queries = query_df[query_df[column].isnull()]["index"]
        if len(empty_queries) == 0:
            continue
        tweet_df = pd.read_csv(tweet_filepath, header=0)
        for i in range(len(empty_queries)):
            tweet_df = tweet_df[tweet_df["index"] != empty_queries.iat[i]]
        query_df = query_df[query_df["title"].notnull()]
        query_df.to_csv(output_filepath, index=False)
        tweet_df.to_csv(output_tweet_filepath, index=False)


def tf_idf(input_dir, output_dir):
    logger.info("Collecting documents for td-idf...")
    all_documents = []  # All documents for tf_idf
    df_lens = []  # Stores dataframe lengths for easy indexing

    for filename in claim_files:
        logger.info(f"- Processing {filename}...")
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath, header=0)
        df_lens.append(len(df))
        all_documents.extend(
            [document for _, document in df["title"].iteritems()])

    for filename in news_files:
        logger.info(f"- Processing {filename}...")
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath, header=0)
        df_lens.append(len(df))
        all_documents.extend(
            [document for _, document in df["concatenated"].iteritems()])

    for filename in tweet_files:
        logger.info(f"- Processing {filename}...")
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath, header=0)
        df_lens.append(len(df))
        all_documents.extend(
            [document for _, document in df["processed"].iteritems()])

    for i in range(len(all_documents)):
        if pd.isnull(all_documents[i]):
            all_documents[i] = ""

    # Determine start and end indices of each csv file
    for i in range(1, 8):
        df_lens[i] += df_lens[i - 1]

    vectorizer = TfidfVectorizer()
    logger.info("Vectorizing...")
    X = vectorizer.fit_transform(all_documents)
    # Size of vocabulary is 45,840
    logger.info("Finished vectorizing. Saving tfidf values...")

    for file_index, filename in enumerate([*claim_files, *news_files, *tweet_files]):
        logger.info(f"- Processing {filename}...")
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath, header=0)
        start = 0 if file_index == 0 else df_lens[file_index - 1]
        end = df_lens[file_index]
        feature_dict_list = []
        for row_index in range(start, end):
            row = coo_matrix(X[row_index])
            feature_dict = {j: v for j, v in zip(row.col, row.data)}
            feature_dict_list.append(feature_dict)
        df["tfidf"] = feature_dict_list
        output_filepath = os.path.join(output_dir, filename)
        df.to_csv(output_filepath, index=False)


def download_glove():
    glove_txt_filename = "glove.twitter.27B.50d.txt"
    glove_txt_path = os.path.join(glove_dir, glove_txt_filename)
    if not os.path.exists(glove_txt_path):
        logger.info("Downloading GloVe word embeddings...")
        glove_zip_url = "https://nlp.stanford.edu/data/glove.twitter.27B.zip"
        downloaded = requests.get(glove_zip_url, allow_redirects=True)
        glove_zip_path = os.path.join(glove_dir, "glove.twitter.27B.zip")
        with open(glove_zip_path, "wb") as f:
            f.write(downloaded.content)
        logger.info("Unzipping...")
        with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
            zip_ref.extractall(glove_dir)
        logger.info("Deleting unused embeddings...")
        for filename in os.listdir(glove_dir):
            if filename != glove_txt_filename:
                os.remove(os.path.join(glove_dir, filename))


def glove(input_dir, output_dir):
    glove_input_filename = "glove.twitter.27B.50d.txt"
    glove_input_path = os.path.join(glove_dir, glove_input_filename)
    glove_w2v_filename = f"{os.path.splitext(glove_input_filename)[0]}.word2vec.txt"
    glove_w2v_path = os.path.join(glove_dir, glove_w2v_filename)
    if not os.path.exists(glove_w2v_path):
        logger.info("Translating from GloVe to Word2Vec...")
        glove2word2vec(glove_input_path, glove_w2v_path)
    logger.info("Loading GloVe model...")
    model = KeyedVectors.load_word2vec_format(glove_w2v_path, binary=False)
    logger.info("Loading done.")

    def get_doc_embedding(document):
        if pd.isnull(document):
            return []
        document = [word for word in document if word in model.vocab]
        if len(document) == 0:
            return []
        mean = np.mean(model[document], axis=0)
        return mean.tolist()

    def add_embeddings(filenames, column):
        for filename in filenames:
            logger.info(f"- Adding document embeddings to {filename}...")
            filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, filename)
            df = pd.read_csv(filepath, header=0)
            embeddings_list = [get_doc_embedding(
                document) for _, document in df[column].iteritems()]
            df["embeddings"] = embeddings_list
            df.to_csv(output_filepath, index=False)

    add_embeddings(claim_files, "title")
    add_embeddings(news_files, "concatenated")
    add_embeddings(tweet_files, "processed")


def make_labels_unique(input_dir, output_dir):
    logger.info("Modifying labels so that they are unique...")
    pairs = zip([*claim_files, *news_files], tweet_files)
    start = 0
    for query_file, tweet_file in pairs:
        query_file_path = os.path.join(input_dir, query_file)
        tweet_file_path = os.path.join(input_dir, tweet_file)
        query_df = pd.read_csv(query_file_path, header=0)
        tweet_df = pd.read_csv(tweet_file_path, header=0)
        label_dict = {k: v for k, v in zip(
            query_df["index"].iteritems(), range(start, start + len(query_df)))}
        query_df["index"] = label_dict.values()
        tweet_df["index"] = tweet_df["index"].replace(label_dict)
        output_query_file_path = os.path.join(output_dir, query_file)
        output_tweet_file_path = os.path.join(output_dir, tweet_file)
        query_df.to_csv(output_query_file_path, index=False)
        tweet_df.to_csv(output_tweet_file_path, index=False)
        start += len(query_df)


def concat_tweet_data(input_dir, output_dir):
    logger.info("Concatenating tweet dataframes...")
    dfs = [pd.read_csv(os.path.join(input_dir, filename), header=0)
           for filename in tweet_files]
    df = pd.concat(dfs)
    output_file_path = os.path.join(output_dir, "all_data.csv")
    df.to_csv(output_file_path, index=False)


class QueryLIBSVM:
    def __init__(self, query_id):
        self.id = query_id
        self.relevant = []
        self.irrelevant = []

    def add_relevant(self, rel_list):
        self.relevant.append(rel_list)

    def add_irrelevant(self, irrel_list):
        self.irrelevant.append(irrel_list)

    def __iter__(self):
        for d in self.relevant:
            yield d, 1
        for d in self.irrelevant:
            yield d, 0

    def __len__(self):
        return len(self.relevant) + len(self.irrelevant)


def build_libsvm_input(input_dir, output_dir):
    input_file_path = os.path.join(input_dir, "all_data.csv")
    df = pd.read_csv(input_file_path, header=0)
    logger.info("Building query dictionary...")
    query_libsvm_dict = defaultdict(lambda: QueryLIBSVM(None))
    index_col, embeddings_col = df.columns.get_loc(
        "index"), df.columns.get_loc("embeddings")
    logger.info("Adding relevant data...")
    for i in range(len(df)):
        embeddings_list = literal_eval(df.iat[i, embeddings_col])
        if len(embeddings_list) == 0:
            continue
        index = df.iat[i, index_col]
        query_libsvm_dict[index].id = index
        query_libsvm_dict[index].add_relevant(embeddings_list)
    logger.info("Adding irrelevant data...")
    to_delete = []
    for query_id in query_libsvm_dict:
        rel_len = len(query_libsvm_dict[query_id].relevant)
        if rel_len < 10:
            to_delete.append(query_id)
            continue
        irrelevant = df[df["index"] != query_id].sample(
            n=rel_len * 2, random_state=seed, ignore_index=False)
        for _, irrel_list in irrelevant["embeddings"].iteritems():
            query_libsvm_dict[query_id].add_irrelevant(
                literal_eval(irrel_list))
    for query_id in to_delete:
        del query_libsvm_dict[query_id]

    query_embeddings_dict = {}
    for filename in [*claim_files, *news_files]:
        temp_path = os.path.join(input_dir, filename)
        temp_df = pd.read_csv(temp_path, header=0, usecols=[
                              "index", "embeddings"])
        for i in range(len(temp_df)):
            query_embeddings_dict[temp_df.iat[i, 0]
                                  ] = literal_eval(temp_df.iat[i, 1])

    logger.info("Creating full dataset...")
    X = []
    y = []
    for query_libsvm in query_libsvm_dict.values():
        qid = query_libsvm.id
        for vector, label in query_libsvm:
            if len(vector) == 0:
                continue
            query_vector = query_embeddings_dict[qid]
            row = {"qid": qid}
            row.update(
                {f"feature{i + 1}": v for i, v in enumerate(query_vector)})
            row.update(
                {f"feature{i + len(query_vector) + 1}": v for i, v in enumerate(vector)})
            X.append(row)
            y.append(label)

    full_X = pd.DataFrame(X)
    full_y = pd.Series(y, name="label")

    logger.info("Splitting data into train and test/validation sets...")
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        full_X, full_y, test_size=0.3, random_state=seed, stratify=full_X["qid"])

    X_test, X_val, y_test, y_val = train_test_split(
        X_remaining, y_remaining, test_size=0.5, random_state=seed, stratify=X_remaining["qid"])

    def libsvm_output(X, y):
        for vector_tuple, label_tuple in zip(X.iterrows(), y.iteritems()):
            (_1, vector), (_2, label) = vector_tuple, label_tuple
            yield f"{label} qid:{vector['qid'].astype(int)} {' '.join([f'{k + 1}:{v}' for k, v in enumerate(list(vector.iloc[1:].astype(str)))])}\n"

    logger.info("Writing to files...")
    # logger.info("- Writing training data...")
    # pd.concat([y_train, X_train], axis=1).to_csv(
    #     os.path.join(output_dir, "input_train.csv"), index=False)
    # logger.info("- Writing test data...")
    # pd.concat([y_test, X_test], axis=1).to_csv(
    #     os.path.join(output_dir, "input_test.csv"), index=False)

    with open(os.path.join(output_dir, "input_train.txt"), "w") as f:
        logger.info("- Writing training data...")
        for line in libsvm_output(X_train, y_train):
            f.write(line)
    with open(os.path.join(output_dir, "input_val.txt"), "w") as f:
        logger.info("- Writing validation data...")
        for line in libsvm_output(X_val, y_val):
            f.write(line)
    with open(os.path.join(output_dir, "input_test.txt"), "w") as f:
        logger.info("- Writing test data...")
        for line in libsvm_output(X_test, y_test):
            f.write(line)

    # count = 0
    # with open(os.path.join(output_dir, "input_train.txt"), "w") as f:
    #     logger.info("- Writing training data...")
    #     for item in query_libsvm_dict.values():
    #         f.write(str(item))
    #         if count == 100:
    #             break
    #         count += 1


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    # copy(processed_tweets_dir, tfidf_dir)
    # concatenate_news(tfidf_dir, tfidf_dir)
    # delete_empty_queries_all(tfidf_dir, tfidf_dir)
    # tf_idf(tfidf_dir, tfidf_dir)

    # copy(processed_tweets_dir, embeddings_dir)
    # concatenate_news(embeddings_dir, embeddings_dir)
    # delete_empty_queries_all(embeddings_dir, embeddings_dir)
    download_glove()
    # glove(embeddings_dir, embeddings_dir)
    # make_labels_unique(embeddings_dir, embeddings_dir)
    # concat_tweet_data(embeddings_dir, embeddings_dir)
    # build_libsvm_input(embeddings_dir, model_input_dir)
