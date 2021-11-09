import os
import logging

import pandas as pd

from .constants import RAW_TWEETS_DIR_NAME

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
raw_tweets_dir = os.path.join(current_dir, RAW_TWEETS_DIR_NAME)


def process(tweet):
    pass


def process_all(raw_tweets_dir=raw_tweets_dir):
    for dirpath, dirnames, filenames in os.walk(raw_tweets_dir):
        pass


if __name__ == "__main__":
    pass
