#importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from utils import TweetProcessor
from imblearn.pipeline import make_pipeline


#loading the dataset after converting tsv into csv
train_data = pd.read_csv(r"C:\Users\rimat\OneDrive\Desktop\final nlp\dataset (OLID)\olid-training.csv")

train_data

# Create an instance of TweetProcessor
tweet_processor = TweetProcessor()

# pre Processing tweets
tfidf_matrix, feature_names = tweet_processor.process_tweets(train_data)

# Now you can use tfidf_matrix and feature_names as needed
#print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
#print("Feature Names:", feature_names)

X = train_data['lemmatized_tweet']
y = train_data['subtask_a']
#splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the Pandas Series to a NumPy array
# Convert the Pandas Series to a NumPy array of strings
X_train_array = X_train.astype(str).values

# Reshape your input data from 1D to 2D
X_train_reshaped = X_train_array.reshape(-1, 1)

#model a pipeline
print(train_data.head())

# Create a pipeline for preprocessing and modeling
modeling_pipeline = make_pipeline(
    RandomOverSampler(random_state=42),  # Over-sampling
    TfidfVectorizer(max_features=100),   # Vectorization
    RandomForestClassifier()            # Classification
)

# Now, you can fit the pipeline to your training data
modeling_pipeline.fit(X_train_reshaped, y_train)

# Make predictions on the test data
predictions = modeling_pipeline.predict(X_test)
