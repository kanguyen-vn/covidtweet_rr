from time import time
import logging
import sys

# from data.dump_raw_tweets import dump_raw_tweets_in_chunks
from data.preprocessing import preprocess_all_separately, concat
from demo.tf_ranking_demo import predict

# Set up logger for project
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("data.log")
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s %(name)s [%(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# dump_raw_tweets_in_chunks()

# preprocess_all_separately()

# concat()

# predictions = predict("drinking bleach")
# print(next(predictions))
