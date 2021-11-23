from .constants import RAW_TWEETS_DIR_NAME, ERROR_IDS_NAME, PROCESSED_TWEETS_DIR_NAME
import os
import logging
import re
import pandas as pd
from string import punctuation
from ast import literal_eval
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import VERB, NOUN, ADJ, ADV
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
import preprocessor as prep
from ekphrasis.classes.segmenter import Segmenter


logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
raw_tweets_dir = os.path.join(current_dir, RAW_TWEETS_DIR_NAME)
processed_tweets_dir = os.path.join(current_dir, PROCESSED_TWEETS_DIR_NAME)

try:
    os.mkdir(processed_tweets_dir)
    logger.info(f"- Creating {processed_tweets_dir} directory.")
except FileExistsError:
    logger.info(f"- {processed_tweets_dir} already created.")

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


def preprocess(tweet, hashtags):
    """Preprocess a tweet."""
    processed = tweet
    # Extract hashtags
    # hashtags = re.findall(r"#(\w+)", tweet)

    # Change whitespaces to spaces
    # processed = " ".join(tweet.split())

    # Segment hashtag
    # processed = re.sub(r"#", "", processed)
    # for hashtag in hashtags:
    #     processed = processed.replace(hashtag, seg_tw.segment(hashtag))

    # Segment hashtags
    hashtags.sort(key=lambda d: d["indices"][1], reverse=True)
    for hashtag in hashtags:
        # processed = processed.replace(hashtag, seg_tw.segment(hashtag))
        start, end = hashtag["indices"]
        processed = processed[:start + 1] + \
            seg_tw.segment(hashtag["text"]) + processed[end:]
    processed = re.sub(r"#", "", processed)

    # Preprocess using preprocessor
    prep.set_options(prep.OPT.URL, prep.OPT.MENTION, prep.OPT.RESERVED,
                     prep.OPT.EMOJI, prep.OPT.SMILEY, prep.OPT.NUMBER)
    processed = prep.clean(processed)

    # Tokenize
    tokens = tk.tokenize(processed)

    # Remove single-letter strings
    tokens = [token for token in tokens if len(token) > 1]

    # Remove stopwords and punctuation
    tokens = [
        token for token in tokens if token not in stops and token not in punctuation]

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


def preprocess_all(raw_tweets_dir=raw_tweets_dir, processed_tweets_dir=processed_tweets_dir):
    """Preprocess all chunks in raw tweets folder."""
    files = [file for file in os.listdir(raw_tweets_dir)
             if os.path.isfile(os.path.join(raw_tweets_dir, file))
             and file != ERROR_IDS_NAME]  # Get only data files in the directory
    for file in files:
        logger.info(f"Preprocessing {file}...")
        raw_chunk_path = os.path.join(raw_tweets_dir, file)
        chunk_path = os.path.join(processed_tweets_dir, file)
        df = pd.read_csv(
            raw_chunk_path,
            header=0,
            dtype=str,
        )
        df.rename({"entities": "hashtags"})
        df["processed"] = ""
        for i in range(len(df)):
            hashtags = literal_eval(df.iat[i, 5])["hashtags"]
            df.iat[i, 5] = hashtags
            df.iat[i, 7] = preprocess(df.iat[i, 2], hashtags)
        df.to_csv(chunk_path, index=False)


def concat(data_dir=processed_tweets_dir):
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
