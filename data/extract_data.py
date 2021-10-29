import os

import pandas as pd
import random

from .constants import FULL_DATASET_NAME, PARTIAL_DATASET_NAME

INITIAL_PERCENTAGE = 0.01
INITIAL_SEED = 27

current_dir = os.path.dirname(os.path.abspath(__file__))

input_path = os.path.join(current_dir, FULL_DATASET_NAME)
output_path = os.path.join(current_dir, PARTIAL_DATASET_NAME)


def sample(input_path=input_path, output_path=output_path, percentage=INITIAL_PERCENTAGE, seed=INITIAL_SEED):
    random.seed(seed)
    df = pd.read_csv(
        os.path.join(current_dir, input_path),
        header=0,
        skiprows=lambda x: x > 0 and random.random() > percentage
    )
    df.to_csv(os.path.join(current_dir, output_path))


def line_count(input_path):
    # There are 198,378,185 lines (including the header) in the full dataset
    count = 0
    with open(input_path) as f:
        for _ in f:
            count += 1
    return count


if __name__ == "__main__":
    sample()
