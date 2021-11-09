import time
import logging
import sys

from data.extract_data import sample
from data.dump_raw_tweets import dump_raw_tweets_in_chunks

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
sample()

start_time = time.time()
dump_raw_tweets_in_chunks()
end_time = time.time()
logger.info(f"Total time taken to get raw tweets: {(end_time - start_time)}")
