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
