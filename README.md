# COVID Twitter Misinformation Lookup: An Investigation of Relevance Ranking Methods

#### Team members: Swati Adhikari, Provakar Mondal, Kiet Nguyen

This repository contains the code for our CS 5604 (Information Storage and Retrieval) final project at Virginia Tech.

Semester: Fall 2021

Instructor: Dr. Ismini Lourentzou

In this course project, we aim to build a search engine using COVID-19-related Tweets as the input data; we experiment with 3 different models: Okapi BM25, RankNet, and TF-Ranking.

## Dataset

The dataset we use for this project is borrowed from this paper:

```
@misc{dharawat2020drink,
      title={Drink bleach or do what now? Covid-HeRA: A dataset for risk-informed health decision making in the presence of COVID19 misinformation},
      author={Arkin Dharawat and Ismini Lourentzou and Alex Morales and ChengXiang Zhai},
      year={2020},
      eprint={2010.08743},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Click [here](data/README.md) for more detailed instructions on downloading the data.

## Requirements

The requirements were generated with `pip freeze > requirements.txt` and can be found in [requirements.txt](requirements.txt).

## Getting data and training models

1. Create a new virtual environment and install the packages noted in [requirements.txt](requirements.txt).
2. Set up your Twitter API keys (as detailed [here](data/README.md)).
3. Run [get_data.sh](get_data.sh) to obtain raw tweets with the Twitter API.
4. Run [prepare_data.sh](prepare_data.sh) to prepare input data for the 3 different models.
5. Run any of the three files in [models](models/) with `python/python3 {filename}` to run or train our models.
