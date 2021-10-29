import os
import sys
import json
import logging
from dotenv import load_dotenv

import pandas as pd
import tweepy

from .constants import RAW_TWEETS_PATH, PARTIAL_DATASET_PATH


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def get_status(api, tweet_id):
    tweet_id = tweet_id.strip()
    tweet_returned = False
    dataframe_row = {}

    try:
        tweet_info = api.get_status(
            tweet_id, tweet_mode="extended")  # get full tweet
        tweet_returned = True
        status_json = json.loads(json.dumps(tweet_info._json))
        dataframe_row = dict([(k, json.dumps(status_json[k]))
                              for k in status_json])
        return tweet_returned, dataframe_row

    except Exception as e:
        if isinstance(e, tweepy.error.TweepError):
            logger.info(f"API Error on id {tweet_id}. Error: {e}")

    return tweet_returned, dataframe_row


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    load_dotenv()  # load env variables in .env file
    consumer_key = os.getenv("OATH_CONSUMER_KEY")
    consumer_secret = os.getenv("OATH_CONSUMER_SECRET")

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    api = tweepy.API(auth)
    # RAW_TWEETS_PATH = "COVID19_twitter_raw_tweets.csv"
    if os.path.exists(RAW_TWEETS_PATH):
        df = pd.read_csv(RAW_TWEETS_PATH)
    else:
        df = pd.DataFrame(columns=["id", "raw_text"])


if __name__ == "__main__":
    main()
    # load_dotenv()  # load env variables in .env file
    # consumer_key = os.getenv("OAUTH_CONSUMER_KEY")
    # consumer_secret = os.getenv("OAUTH_CONSUMER_SECRET")

    # auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # api = tweepy.API(auth)
    # print(json.loads(json.dumps(api.get_status(
    #     "1221957211913457664", tweet_mode="extended")._json)))
