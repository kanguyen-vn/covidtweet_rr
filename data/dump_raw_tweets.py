import os
import json
import logging
import time
import datetime
from dotenv import load_dotenv

import pandas as pd
import tweepy

from .constants import RAW_TWEETS_DIR_NAME, PARTIAL_DATASET_NAME

logger = logging.getLogger(__name__)

API_RATE_LIMIT = 900  # rate limit set
SLEEP_TIME = 60 * (15 + 1)  # seconds. Sleep for 16 minutes
ERROR_IDS_NAME = "error_ids.csv"

logger.info("Reading consumer keys for Twitter API...")
load_dotenv()  # load env variables in .env file
consumer_key = os.getenv("OAUTH_CONSUMER_KEY")
consumer_secret = os.getenv("OAUTH_CONSUMER_SECRET")
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)

logger.info("Setting up directories...")
current_dir = os.path.dirname(os.path.abspath(__file__))
raw_tweets_dir = os.path.join(current_dir, RAW_TWEETS_DIR_NAME)
try:
    os.mkdir(raw_tweets_dir)
    logger.info(f"- Creating {raw_tweets_dir} directory.")
except FileExistsError:
    logger.info(f"- {raw_tweets_dir} already created.")
input_file = os.path.join(current_dir, PARTIAL_DATASET_NAME)


def sleep():
    logger.info(f"Sleeping for {SLEEP_TIME / 60} minutes...")
    for i in range(SLEEP_TIME, 0, -1):
        print(
            f"Time remaining: {datetime.timedelta(seconds=i)}",
            end="\r",
            flush=True
        )
        time.sleep(1)


def get_latest_tweet_id(raw_tweets_dir=raw_tweets_dir):
    """Get latest tweet ID that was processed."""
    files = [file for file in os.listdir(raw_tweets_dir)
             if os.path.isfile(os.path.join(raw_tweets_dir, file))
             and file != ERROR_IDS_NAME]  # Get only data files in the directory
    if len(files) == 0:  # If no chunks are available, check error file
        if os.path.exists(os.path.join(raw_tweets_dir, ERROR_IDS_NAME)):
            latest = ERROR_IDS_NAME
        else:
            return None
    else:
        files = [int(os.path.splitext(file)[0])
                 for file in files]  # Get ints only from file names
        latest = str(max(files)) + ".csv"
    df = pd.read_csv(
        os.path.join(raw_tweets_dir, latest),
        header=0,
        usecols=["tweet_id"],
        dtype=str
    )
    return df.iat[-1, 0] if not df.empty else None


def get_text(api, tweet_id):
    """Get tweet text by tweet_id using the Twitter API."""
    tweet_id = tweet_id.strip()
    tweet_returned = False
    text = ""
    exception = None

    try:
        tweet_info = api.get_status(
            tweet_id, tweet_mode="extended")  # get full tweet
        tweet_returned = True
        status_json = json.loads(json.dumps(tweet_info._json))
        text = status_json["full_text"]

    except Exception as e:
        exception = e
        if isinstance(e, tweepy.TweepyException):
            logger.error(f"API error on id {tweet_id}, error: {e}")

    return tweet_returned, text, exception


def dump_raw_tweets_in_chunks(api=api, input_file=input_file, raw_tweets_dir=raw_tweets_dir, chunk_size=5000):
    """Dump raw tweet texts in separate files (chunks), each containing chunk_size rows or tweets."""
    input_df = pd.read_csv(
        input_file,
        header=0,
        usecols=["tweet_id"],
        dtype=str
    )
    num_rows = len(input_df)
    num_chunks = num_rows // chunk_size + 1

    # Set up dataframe to store ids with errors
    error_ids_path = os.path.join(raw_tweets_dir, ERROR_IDS_NAME)
    if os.path.exists(error_ids_path):
        error_df = pd.read_csv(error_ids_path, header=0, dtype=str)
    else:
        error_df = pd.DataFrame(columns=["tweet_id", "error_type"])

    # Get latest tweet_id so as not to start over
    latest_tweet_id = get_latest_tweet_id(raw_tweets_dir)
    latest_index = int(
        input_df.index[input_df["tweet_id"] == latest_tweet_id][0]) if latest_tweet_id else 0

    try:
        # Iterate indices by chunk (file)
        for chunk_index in range(0, num_rows, chunk_size):
            chunk_start = chunk_index
            chunk_end = min(chunk_index + chunk_size, num_rows)

            if chunk_end < latest_index:
                continue

            file_no = chunk_index // chunk_size
            chunk_path = os.path.join(raw_tweets_dir, f"{file_no}.csv")
            logger.info(f"Dumping raw tweets in chunk {file_no}/{num_chunks}:")
            df = pd.DataFrame(columns=["tweet_id", "raw_text"])

            # Iterate indices in each chunk by rate limit
            for limit_index in range(chunk_start, chunk_end, API_RATE_LIMIT):
                start = limit_index
                end = min(limit_index + API_RATE_LIMIT, chunk_end)

                if end < latest_index:
                    continue

                logger.info(
                    f"- Dumping indices from {start} to {end}...")

                # Still O(n)
                for index in range(start, end):
                    if index < latest_index:
                        continue
                    tweet_id = input_df.iat[index, 0]
                    if tweet_id in error_df.tweet_id:
                        continue
                    retry = True
                    while retry:
                        tweet_returned, text, exception = get_text(
                            api, tweet_id)
                        if tweet_returned:
                            row = {"tweet_id": tweet_id, "raw_text": text}
                            df = df.append(row, ignore_index=True)
                            retry = False
                            continue
                        if isinstance(exception, tweepy.TooManyRequests):  # Rate limit hit?
                            logger.error(
                                "Rate limit hit exception. Taking a break...")
                            sleep()
                        else:
                            error_row = {"tweet_id": tweet_id,
                                         "error_type": type(exception).__name__}
                            error_df = error_df.append(
                                error_row, ignore_index=True)
                            retry = False
                logger.info(
                    "Rate limit hit (no exception). Printing current progress and taking a break...")
                df.to_csv(chunk_path, index=False)
                error_df.to_csv(error_ids_path, index=False)
                sleep()

            df.to_csv(chunk_path, index=False)
            error_df.to_csv(error_ids_path, index=False)
            logger.info("Pausing between chunks...")
            sleep()
    except Exception as e:
        # Any other type of error, then shut down
        if isinstance(e, tweepy.TooManyRequests):
            logger.error("Rate limit error")
        else:
            logger.error(f"Encountered exception {e}")
        logger.info("Closing API")
        return  # Exit condition


if __name__ == "__main__":
    start_time = time.time()
    dump_raw_tweets_in_chunks()
    end_time = time.time()
    logger.info(f"Total time taken: {(end_time - start_time)}")
