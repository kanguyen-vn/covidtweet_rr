import pandas as pd
import random

from .constants import FULL_DATASET_PATH, PARTIAL_DATASET_PATH

INITIAL_PERCENTAGE = 0.01
INITIAL_SEED = 27


def sample(input_path=FULL_DATASET_PATH, output_path=PARTIAL_DATASET_PATH, percentage=INITIAL_PERCENTAGE, seed=INITIAL_SEED):
    random.seed(seed)
    df = pd.read_csv(
        input_path,
        header=0,
        skiprows=lambda x: x > 0 and random.random() > percentage
    )
    df.to_csv(output_path)


def line_count(input_path):
    # There are 198,378,185 lines (including the header) in the full dataset
    count = 0
    with open(input_path) as f:
        for _ in f:
            count += 1
    return count


if __name__ == "__main__":
    sample()
