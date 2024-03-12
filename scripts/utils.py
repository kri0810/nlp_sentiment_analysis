import pandas as pd
import string
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import string
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import string
import re

class TweetProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        pass

    def clean_tweet(self, tweet):
        punctuations = string.punctuation
        stop_words = set(stopwords.words('english'))

        # Remove mentions, URLs, and special characters
        tweet = tweet.replace('@USER', '') \
                     .replace('URL', '') \
                     .replace('&amp', 'and') \
                     .replace('&lt','') \
                     .replace('&gt','') \
                     .replace('\d+','') \
                     .lower()

        # Remove punctuations
        for punctuation in punctuations:
            tweet = tweet.replace(punctuation, '')

        # Remove emojis
        tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)

        # Remove stopwords
        tweet = ' '.join([word for word in tweet.split() if word not in stop_words])

        # Trim leading and trailing whitespaces
        tweet = tweet.strip()

        return tweet
    
        pass
    

    def tokenize_tweet(self, tweet):
        return word_tokenize(tweet)

    def lemmatize_tweet(self, tokens):
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens
    
        pass
    
    def process_tweets(self, df, ngram_range=(1, 3), stop_words='english'):
        # Clean tweets
        df['cleaned_tweet'] = df['tweet'].apply(self.clean_tweet)

        # Tokenize tweets
        df['tokenized_tweet'] = df['cleaned_tweet'].apply(self.tokenize_tweet)

        # Lemmatize tweets
        df['lemmatized_tweet'] = df['tokenized_tweet'].apply(self.lemmatize_tweet)

        # Join lemmatized tokens back into strings
        df['lemmatized_tweet'] = df['lemmatized_tweet'].apply(lambda x: ' '.join(x))

        # Vectorize tweets using TF-IDF
        tfidf_matrix, feature_names = self.vectorize_tweets(df, stop_words=stop_words, ngram_range=ngram_range)

        return tfidf_matrix, feature_names
    
        pass


    def vectorize_tweets(self, df, stop_words=None, ngram_range=(1, 1), column_name='lemmatized_tweet'):
        # Initialize TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=ngram_range)

        # Fit and transform the lemmatized tweets
        tfidf_matrix = tfidf_vectorizer.fit_transform(df[column_name])

        return tfidf_matrix, tfidf_vectorizer.get_feature_names_out()