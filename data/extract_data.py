from json.decoder import JSONDecodeError
import os
import logging
import json
import random

import pandas as pd

FULL_DATASET_NAME = "COVID19_twitter_full_dataset.csv"
PARTIAL_DATASET_NAME = "COVID19_twitter_partial_dataset.csv"
PARTIAL_DATASET_METADATA_NAME = "partial_dataset_metadata.json"
SAMPLE_FRACTION = 0.001
INITIAL_SEED = 27

current_dir = os.path.dirname(os.path.abspath(__file__))

input_path = os.path.join(current_dir, FULL_DATASET_NAME)
output_path = os.path.join(current_dir, PARTIAL_DATASET_NAME)
metadata_path = os.path.join(current_dir, PARTIAL_DATASET_METADATA_NAME)

logger = logging.getLogger(__name__)


def sample(input_path=input_path, output_path=output_path, fraction=SAMPLE_FRACTION, seed=INITIAL_SEED, metadata_path=metadata_path):
    """Sample from full dataset if haven't already, and return length of sample."""
    if not os.path.exists(output_path):
        logger.info(
            f"- Sampling {fraction * 100}% from {output_path}...")
        random.seed(seed)
        df = pd.read_csv(
            os.path.join(current_dir, input_path),
            header=0,
            usecols=["tweet_id", "tweet_timestamp", "keyword"],
            skiprows=lambda x: x > 0 and random.random() > fraction,
            dtype=str
        )
        df.to_csv(os.path.join(current_dir, output_path), index=False)
        logger.info(f"- Sampling done, output in {output_path}.")
        metadata = {
            "path": output_path,
            "length": len(df),
            "fraction": fraction,
            "seed": seed
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        return len(df)
    else:
        logger.info(f"- {output_path} already exists.")
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except JSONDecodeError:
            metadata = {
                "path": output_path,
                "length": line_count(output_path),
                "fraction": fraction,
                "seed": seed
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

        return metadata["length"]


def line_count(input_path):
    """Count number of lines in a CSV file."""
    # There are 198,378,185 lines (including the header) in the full dataset
    count = 0
    with open(input_path) as f:
        possible_header = f.readline()
        has_header = possible_header[0] not in "-.0123456789"
        for _ in f:
            count += 1
    return count - 1 if has_header else count


if __name__ == "__main__":
    sample()
