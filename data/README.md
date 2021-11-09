## Data Extraction

The code in this directory can be used to download and preprocess text data retrieved by the Twitter API.

### Setting up the API

Place your Twitter API credentials in an `.env.` file in the root directory (`..` from this directory). The `.env.` file should look like this:

```
OAUTH_CONSUMER_KEY=*YOUR_TWITTER_API_CONSUMER_KEY*
OAUTH_CONSUMER_SECRET=*YOUR_TWITTER_API_CONSUMER_SECRET*
```

Note that there should NOT be any quotation marks or apostrophes surrounding the credentials. Learn more about the Twitter API [here](https://developer.twitter.com/en/docs/twitter-api).

### Download the dataset

The dataset we use for this project is borrowed from this paper:

> Raj Gupta, Ajay Vishwanath, Yinping Yang. [“COVID-19 Twitter Dataset with Latent Topics, Sentiments and Emotions Attributes.”](https://arxiv.org/pdf/2007.06954.pdf) _Projects: Emotional Responses surrounding COVID-19, September 2020_. Lab: Digital Emotion and Empathy Machine.

The full dataset is available for download at the [OpenICPSR COVID-19 Data Repository](https://doi.org/10.3886/E120321).

The full dataset (`COVID19_twitter_full_dataset.csv`) is rather heavy (22.1GB) so we want to sample from this data. Place the full dataset in the current `data` directory. Executing `extract_data.sample()` will generate a partial dataset that is 0.1% the size of the full dataset and place the resulting file (`COVID19_twitter_partial_dataset.csv`) in the same directory.
