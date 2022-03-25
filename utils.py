import os
import pandas as pd

from sklearn.model_selection import train_test_split

SENTIMENT140 = 1
TWITTER = 2
TWITTER_AIRLINES = 3
TWITTER_APPLE = 4
WNUT_2016 = 5
WNUT_2017 = 6
TWEEBANK = 7

SENTIMENT_PATH = os.path.join('data', 'sentiment')
NER_PATH = os.path.join('data', 'ner')

def dataset_reader(dataset_number):
    """
    
    Parameters:
    dataset_number: int in {1,2,..,7}
    
    1-> Sentiment140
    2-> Twitter Sentiment Analysis
    3-> Twitter Sentiment Analysis on Airlines
    4-> Apple Twitter Sentiment
    5-> WNUT 2016: Emerging and Rare entity recognition
    6-> WNUT 17: Emerging and Rare entity recognition 
    7-> TweeBank v2
    
    Return:
    
    Three pandas dataframse: train, dev, test
    """
    train, validate, test = (None, None, None)    
    
    if dataset_number == 2:
        # Twitter Sentiment Analysis 
        train_p = os.path.join(SENTIMENT_PATH, 'twitter-sentiment-analysis' ,'twitter_training.csv')
        validate_p = os.path.join(SENTIMENT_PATH, 'twitter-sentiment-analysis' ,'twitter_validation.csv')
        
        df = pd.read_csv(train_p)                        
        train = df.sample(frac = 0.7)
        test = df.drop(train.index)
        validate = pd.read_csv(validate_p)
        
    if dataset_number == 3:
        # Twitter Sentiment Analysis on Airlines
        train_p = os.path.join(SENTIMENT_PATH, "twitter-us-airline-sentiment", "Tweets.csv")
        df = pd.read_csv(train_p)
        
        train = df.sample(frac = 0.8)
        non_train = df.drop(train.index)
        validate = non_train.sample(frac = 0.5)
        test = non_train.drop(validate.index)
        
    if dataset_number == 4:
        # Apple Twitter Sentiment
        apple_path = os.path.join(SENTIMENT_PATH, "crowdflower-apple-twitter-sentiment", "data", "apple_twitter_sentiment_dfe.csv")
        df = pd.read_csv(apple_path)
        
        train = df.sample(frac = 0.8)
        non_train = df.drop(train.index)
        validate = non_train.sample(frac = 0.5)
        test = non_train.drop(validate.index)

    return train, validate, test

def prune_columns(ds_num, df):
    """
    Return reformatted dataframes with consistent columns
    index, text, sentiment, [airline]
    """
    if ds_num == 2:
        return df[['topic_id','text','sentiment']]
    
    if ds_num == 3: 
        df = df.rename(columns={'airline_sentiment':'sentiment'})
        return df[['tweet_id','text','sentiment','airline']]
    
    if ds_num == 4:
        df = df.rename(columns={'unit_id':'tweet_id'})
        return df[['tweet_id','text','sentiment']]
