#importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from scripts.utils import TweetProcessor
from imblearn.pipeline import make_pipeline
import pickle


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

df = train_data[["lemmatized_tweet", "subtask_a", "subtask_b", "subtask_c"]]
df['subtask_a'] = df['subtask_a'].map({'OFF': 1, 'NOT': 0})
df['subtask_b'] = df['subtask_b'].map({'TIN': 1, 'UNT': 0})
df['subtask_c'] = df['subtask_c'].map({'OTH':2, 'IND': 1, 'GRP': 0})


#SUBTASK A
X_subtaska = df['lemmatized_tweet']
y_subtaska = df['subtask_a']


# Perform downsampling to balance the classes
rus = RandomOverSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_subtaska.values.reshape(-1, 1), y_subtaska)
# Vectorize the resampled text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features = 100)
X_resampled_tfidf = tfidf_vectorizer.fit_transform(X_resampled.ravel())

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_resampled_tfidf, y_resampled, test_size=0.2, random_state=42)

#loading model
# Load the saved model from file
with open('random_forest_model_a.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# Make predictions on the test data
predictions = loaded_model.predict(X_valid)
print([predictions])

#SUBTASK B
#filtering data for passing to model 2
X_subtaskb = X_subtaska[df['subtask_a'] == 1]
y_subtaskb = df.loc[df['subtask_a'] == 1, 'subtask_b']


# Perform downsampling to balance the classes
rus = RandomOverSampler(random_state=42)
X_resampled_b, y_resampled_b = rus.fit_resample(X_subtaskb.values.reshape(-1, 1), y_subtaskb)

# Vectorize the resampled text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features = 100)
X_resampled_tfidf_b = tfidf_vectorizer.fit_transform(X_resampled_b.ravel())

# Split the data into training and validation sets
X_train_b, X_valid_b, y_train_b, y_valid_b = train_test_split(X_resampled_tfidf_b, y_resampled_b, test_size=0.2, random_state=42)

# Load the saved model from file
with open('random_forest_model_b.pkl', 'rb') as file:
    loaded_model_b = pickle.load(file)

# Make predictions on the test data
predictions = loaded_model_b.predict(X_valid)
print([predictions])

#SUBTASK C

#filtering data for passing to model 3
X_subtaskc = X_subtaskb[df['subtask_b'] == 1]
y_subtaskc = df.loc[df['subtask_b'] == 1, 'subtask_c']


ros_c = RandomOverSampler(random_state=42)
X_resampled_c, y_resampled_c = ros_c.fit_resample(X_subtaskc.values.reshape(-1, 1), y_subtaskc)

print(len(X_resampled_c))
# Vectorize the resampled text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features = 100)
X_resampled_tfidf_c = tfidf_vectorizer.fit_transform(X_resampled_c.ravel())

# Split the data into training and validation sets
X_train_c, X_valid_c, y_train_c, y_valid_c = train_test_split(X_resampled_tfidf_c, y_resampled_c, test_size=0.2, random_state=42)
print(X_valid_c)
# Load the saved model from file
with open('random_forest_model_c.pkl', 'rb') as file:
    loaded_model_c = pickle.load(file)
# Make predictions on the test data
pred= loaded_model_c.predict(X_valid_c)
print([pred])

# Load the unseen data
# Preprocess the unseen data
unseen_tweet = "what are you doing man"
cleaned_unseen_tweet = tweet_processor.clean_tweet(unseen_tweet)

# Vectorize the unseen data using the TF-IDF vectorizer
X_unseen = tfidf_vectorizer.transform([cleaned_unseen_tweet])

# Make predictions using the trained models
prediction_a = loaded_model.predict(X_unseen)
prediction_b = loaded_model_b.predict(X_unseen)
prediction_c = loaded_model_c.predict(X_unseen)

# You can now use the predictions for further analysis or display them as needed
print("Prediction for Subtask A:", prediction_a)
print("Prediction for Subtask B:", prediction_b)
print("Prediction for Subtask C:", prediction_c)
