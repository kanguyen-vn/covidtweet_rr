import os
import logging
import re
import pandas as pd
from shutil import copy2
from string import punctuation
from ast import literal_eval
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import VERB, NOUN, ADJ, ADV
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
import preprocessor as prep
from ekphrasis.classes.segmenter import Segmenter

RAW_TWEETS_DIR_NAME = "raw_tweets"
ERROR_IDS_NAME = "error_ids.csv"
PROCESSED_TWEETS_DIR_NAME = "processed_tweets"

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
raw_tweets_dir = os.path.join(current_dir, RAW_TWEETS_DIR_NAME)
processed_tweets_dir = os.path.join(current_dir, PROCESSED_TWEETS_DIR_NAME)

try:
    os.mkdir(processed_tweets_dir)
except FileExistsError:
    pass

seg_tw = Segmenter(corpus="twitter")
tk = TweetTokenizer(preserve_case=False, reduce_len=True)
# stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stops = stopwords.words("english")
stopwords_path = os.path.join(current_dir, "stopwords.txt")
with open(stopwords_path) as f:
    new_stopwords = f.readlines()
new_stopwords = [i.rstrip("\n") for i in new_stopwords]
new_stopwords = [i for i in new_stopwords if i not in stops]
stops.extend(new_stopwords)


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return ADJ
    elif tag.startswith("V"):
        return VERB
    elif tag.startswith("N"):
        return NOUN
    elif tag.startswith("R"):
        return ADV
    else:
        return None


def preprocess(text, hashtags=None):
    """Preprocess a tweet or some document."""
    processed = text
    # Extract hashtags
    # hashtags = re.findall(r"#(\w+)", tweet)

    # Segment hashtag
    # processed = re.sub(r"#", "", processed)
    # for hashtag in hashtags:
    #     processed = processed.replace(hashtag, seg_tw.segment(hashtag))

    # Segment hashtags
    if hashtags is not None:
        hashtags.sort(key=lambda d: d["indices"][1], reverse=True)
        for hashtag in hashtags:
            # processed = processed.replace(hashtag, seg_tw.segment(hashtag))
            start, end = hashtag["indices"]
            processed = processed[:start + 1] + \
                seg_tw.segment(hashtag["text"]) + processed[end:]
    processed = re.sub(r"#", "", processed)

    # Change whitespaces to spaces
    processed = " ".join([line.strip() for line in processed.splitlines()])
    processed = " ".join(processed.split())

    # Preprocess using preprocessor
    prep.set_options(prep.OPT.URL, prep.OPT.MENTION, prep.OPT.RESERVED,
                     prep.OPT.EMOJI, prep.OPT.SMILEY, prep.OPT.NUMBER)
    processed = prep.clean(processed)

    # Remove punctuation
    processed = "".join(
        [char for char in processed if char not in punctuation])

    # Tokenize
    tokens = tk.tokenize(processed)

    # Remove single-letter strings
    tokens = [token for token in tokens if len(token) > 1]

    # Remove stopwords and punctuation
    tokens = [
        token for token in tokens if token not in stops and token not in punctuation]

    # Remove numbers
    tokens = [token for token in tokens if not token.isdigit()]

    # Lemmatizing
    tagged = pos_tag(tokens)
    for index, pair in enumerate(tagged):
        token, tag = pair
        pos = get_wordnet_pos(tag)
        lemmatized = lemmatizer.lemmatize(
            token, pos=pos) if pos else lemmatizer.lemmatize(token)
        # try
        if tag == "NNS" and lemmatized == token:
            lemmatized = lemmatizer.lemmatize(
                token, pos=VERB)
        tokens[index] = lemmatized

    # Stemming
    # tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)


# columns_to_keep = ["label", "id", "entities", "full_text"]


# def preprocess_all(raw_tweets_dir=raw_tweets_dir, processed_tweets_dir=processed_tweets_dir):
#     """Preprocess all chunks in raw tweets folder."""
#     files = [file for file in os.listdir(raw_tweets_dir)
#              if os.path.isfile(os.path.join(raw_tweets_dir, file))
#              and file != ERROR_IDS_NAME]  # Get only data files in the directory
#     for file in files:
#         logger.info(f"Preprocessing {file}...")
#         raw_chunk_path = os.path.join(raw_tweets_dir, file)
#         chunk_path = os.path.join(processed_tweets_dir, file)
#         df = pd.read_csv(
#             raw_chunk_path,
#             header=0,
#             usecols=columns_to_keep,
#             dtype=str,
#         )
#         df.rename(columns={"entities": "hashtags"}, inplace=True)
#         df["processed"] = ""
#         for i in range(len(df)):
#             # hashtags = literal_eval(df.iat[i, 5])["hashtags"]
#             # df.iat[i, 5] = hashtags
#             # df.iat[i, 7] = preprocess(df.iat[i, 2], hashtags)
#             hashtags = literal_eval(df.iat[i, 2])["hashtags"]
#             df.iat[i, 2] = [hashtag["text"] for hashtag in hashtags]
#             df.iat[i, 4] = preprocess(literal_eval(df.iat[i, 3]), hashtags)
#         df.to_csv(chunk_path, index=False)


def concat(data_dir):
    files = [file for file in os.listdir(data_dir)
             if os.path.isfile(os.path.join(data_dir, file))
             and file != ERROR_IDS_NAME]  # Get only data files in the directory
    files = [int(os.path.splitext(file)[0])
             for file in files]  # Get ints only from file names
    latest = max(files)
    files = []
    for i in range(0, latest + 1):
        files.append(f"{i}.csv")
    df = pd.concat([
        pd.read_csv(
            os.path.join(data_dir, file),
            header=0,
            dtype=str,
        )
        for file in files])
    df.to_csv(os.path.join(data_dir, "all_data.csv"), index=False)


def preprocess_all_separately(raw_tweets_dir=raw_tweets_dir, processed_tweets_dir=processed_tweets_dir):
    for filename in ["ClaimFakeCOVID-19.csv", "ClaimRealCOVID-19.csv"]:
        logger.info(f"Preprocessing {filename}...")
        filepath = os.path.join(raw_tweets_dir, filename)
        df = pd.read_csv(filepath, header=0)
        df.rename(columns={df.columns[0]: "index"}, inplace=True)
        df = df[["index", "title"]]
        for i in range(len(df)):
            df.iat[i, 1] = preprocess(literal_eval(df.iat[i, 1]))
        processed_filepath = os.path.join(processed_tweets_dir, filename)
        df.to_csv(processed_filepath, index=False)

    for filename in ["NewsFakeCOVID-19.csv", "NewsRealCOVID-19.csv"]:
        logger.info(f"Preprocessing {filename}...")
        filepath = os.path.join(raw_tweets_dir, filename)
        df = pd.read_csv(filepath, header=0)
        df.rename(columns={df.columns[0]: "index"}, inplace=True)
        df = df[["index", "title", "newstitle", "content", "abstract"]]
        for i in range(len(df)):
            for j in range(1, 5):
                if pd.isnull(df.iat[i, j]):
                    continue
                try:
                    df.iat[i, j] = literal_eval(df.iat[i, j])
                except:
                    pass
                df.iat[i, j] = preprocess(df.iat[i, j])
        processed_filepath = os.path.join(processed_tweets_dir, filename)
        df.to_csv(processed_filepath, index=False)

    tweet_files = [f"{pref}COVID-19_tweets_expanded.csv" for pref in [
        "ClaimFake", "ClaimReal", "NewsFake", "NewsReal"]]
    for filename in tweet_files:
        logger.info(f"Preprocessing {filename}...")
        filepath = os.path.join(raw_tweets_dir, filename)
        df = pd.read_csv(filepath, header=0)
        df = df[["index", "id", "full_text", "entities",
                 "favorite_count", "retweet_count"]]
        df.rename(columns={"full_text": "processed",
                           "entities": "hashtags"}, inplace=True)
        for i in range(len(df)):
            hashtags = None
            if not pd.isnull(df.iat[i, 3]):
                hashtags = literal_eval(df.iat[i, 3])["hashtags"]
                df.iat[i, 3] = [hashtag["text"] for hashtag in hashtags]
            if pd.isnull(df.iat[i, 2]):
                continue
            try:
                df.iat[i, 2] = literal_eval(df.iat[i, 2])
            except:
                pass
            df.iat[i, 2] = preprocess(df.iat[i, 2], hashtags)
        processed_filepath = os.path.join(processed_tweets_dir, filename)
        df.to_csv(processed_filepath, index=False)

    filepath = os.path.join(raw_tweets_dir, "full_dataset_label_ids.csv")
    copy2(filepath, processed_tweets_dir)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    preprocess_all_separately()
