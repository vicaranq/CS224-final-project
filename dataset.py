import os
import transformers
from transformers import BertModel, BertTokenizer
# import torch
import vsm
import numpy as np
import pandas as pd
# import seaborn as sns
# from pylab import rcParams
import matplotlib.pyplot as plt
# from matplotlib import rc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
# from torch import nn, optim
# from torch.utils.data import Dataset, DataLoader

# sns.set(style='whitegrid', palette='muted', font_scale=1.2)
# HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
# sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
# rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42


SENTIMENT140 = 1
TWITTER = 2
TWITTER_AIRLINES = 3
TWITTER_APPLE = 4
WNUT_2016 = 5
WNUT_2017 = 6
TWEEBANK = 7

SENTIMENT_PATH = os.path.join('data', 'sentiment')
NER_PATH = os.path.join('data', 'ner')

bert_weights_name = 'bert-base-cased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_weights_name)
bert_model = BertModel.from_pretrained(bert_weights_name)
# model = BertForSequenceClassification.from_pretrained(bert_weights_name)

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
    
    if dataset_number == TWITTER:
        # Twitter Sentiment Analysis 
        train_p = os.path.join(SENTIMENT_PATH, 'twitter-sentiment-analysis' ,'twitter_training.csv')
        validate_p = os.path.join(SENTIMENT_PATH, 'twitter-sentiment-analysis' ,'twitter_validation.csv')
        
        df = pd.read_csv(train_p)                        
        train = df.sample(frac = 0.7)
        test = df.drop(train.index)
        validate = pd.read_csv(validate_p)
        
    if dataset_number == TWITTER_AIRLINES:
        # Twitter Sentiment Analysis on Airlines
        train_p = os.path.join(SENTIMENT_PATH, "twitter-us-airline-sentiment", "Tweets.csv")
        df = pd.read_csv(train_p)
        
        train = df.sample(frac = 0.8)
        non_train = df.drop(train.index)
        validate = non_train.sample(frac = 0.5)
        test = non_train.drop(validate.index)
        
    if dataset_number == TWITTER_APPLE:
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
    if ds_num == TWITTER:
        df = df.rename(columns={'topic_id':'entity'})
        return df[['text','sentiment','entity']]
    
    if ds_num == TWITTER_AIRLINES: 
        df = df.rename(columns={'airline_sentiment':'sentiment'})
        return df[['tweet_id','text','sentiment','airline']]
    
    if ds_num == TWITTER_APPLE:
        df = df.rename(columns={'unit_id':'tweet_id'})
        return df[['tweet_id','text','sentiment']]

    
def fit_softmax_classifier(X, y):
    mod = LogisticRegression(
        fit_intercept=True,
        solver='liblinear',
        multi_class='ovr')
    mod.fit(X, y)
    return mod


def hf_cls_phi(text):
    # Get the ids. `vsm.hf_encode` will help; be sure to
    # set `add_special_tokens=True`.
    ##### YOUR CODE HERE
    subtok_ids = vsm.hf_encode(text, bert_tokenizer, add_special_tokens=True)

    # Get the BERT representations. `vsm.hf_represent` will help:
    ##### YOUR CODE HERE
    subtok_reps = vsm.hf_represent(subtok_ids, bert_model, layer=-1)

    # Index into `reps` to get the representation above [CLS].
    # The shape of `reps` should be (1, n, 768), where n is the
    # number of tokens. You need the 0th element of the 2nd dim:
    ##### YOUR CODE HERE
    cls_rep = subtok_reps[0][:][0]

    # These conversions should ensure that you can work with the
    # representations flexibly. Feel free to change the variable
    # name:
    return cls_rep.cpu().numpy()
# def Dataset(Dataset)