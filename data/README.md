## Data Extraction

The code in this directory can be used to download and preprocess text data retrieved by the Twitter API.

### Setting up the API

Place your Twitter API credentials in an `.env` file in the root directory (`..` from this directory). The `.env` file should look like this:

<pre>
OAUTH_CONSUMER_KEY=<i>YOUR_TWITTER_API_CONSUMER_KEY</i>
OAUTH_CONSUMER_SECRET=<i>YOUR_TWITTER_API_CONSUMER_SECRET</i>
</pre>

Note that there should NOT be any quotation marks or apostrophes surrounding the credentials. Learn more about the Twitter API [here](https://developer.twitter.com/en/docs/twitter-api).

### Download the dataset

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

### GloVe word embeddings

For this project, we use the 50-dimensional GloVe word embeddings to vectorize our text data, where each document vector is the average of all its term vectors.

```
@inproceedings{pennington2014glove,
  author = {Jeffrey Pennington and Richard Socher and Christopher D. Manning},
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  title = {GloVe: Global Vectors for Word Representation},
  year = {2014},
  pages = {1532--1543},
  url = {http://www.aclweb.org/anthology/D14-1162},
}
```

Please download `glove.twitter.27B.zip` from [here](https://nlp.stanford.edu/projects/glove/), then extract it to get `glove.twitter.27B.50d.txt` and place it in a folder named _glove_ in the base directory.
