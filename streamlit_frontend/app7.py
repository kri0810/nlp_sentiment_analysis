import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import TweetProcessor

def load_models():
    try:
        with open('random_forest_model_a.pkl', 'rb') as file:
            loaded_model_a = pickle.load(file)
        
        with open('random_forest_model_b.pkl', 'rb') as file:
            loaded_model_b = pickle.load(file)
        
        with open('random_forest_model_c.pkl', 'rb') as file:
            loaded_model_c = pickle.load(file)
        
        return loaded_model_a, loaded_model_b, loaded_model_c
    except Exception as e:
        st.error(f"Error occurred while loading models: {e}")
        return None, None, None

# Function to make predictions
def make_predictions(unseen_tweet, tfidf_vectorizer, loaded_model_a, loaded_model_b, loaded_model_c):
    try:
        # Vectorize the unseen data using the TF-IDF vectorizer
        X_unseen = tfidf_vectorizer.transform([unseen_tweet])

        # Empty list to store prediction results
        predictions = []

        # Make predictions using Model A
        prediction_a = loaded_model_a.predict(X_unseen)
        if prediction_a == 0:
            predictions.append("Not offensive")
        else:
            predictions.append("Offensive")

            # Make predictions using Model B
            prediction_b = loaded_model_b.predict(X_unseen)
            
            # Check if the tweet is targeted
            if prediction_b == 0:
                predictions.append("Untargeted")
            else:
                predictions.append("Targeted")

                # Make predictions using Model C
                prediction_c = loaded_model_c.predict(X_unseen)
                if prediction_c == 0:
                    predictions.append('Group') 
                elif prediction_c == 1:
                    predictions.append('Individual') 
                else:
                    predictions.append('Others') 

        return predictions
    except Exception as e:
        st.error(f"Error occurred while making predictions: {e}")
        return []

# Streamlit UI
def main():
    st.title('Tweet Offensiveness Prediction')

    # Sidebar for entering unseen tweet
    unseen_tweet = st.text_input('Enter the tweet:')

    # Creating an instance of TweetProcessor
    tweet_processor = TweetProcessor()

    # Preprocessing the unseen tweet
    try:
        preprocessed_tweet = tweet_processor.process_tweets(pd.DataFrame({'tweet': [unseen_tweet]}))['stemmed_tweet'].iloc[0]
    except Exception as e:
        st.error(f"Error occurred while preprocessing tweet: {e}")
        preprocessed_tweet = ''

    # Loading the trained models
    loaded_model_a, loaded_model_b, loaded_model_c = load_models()

    # Button to make predictions
    if st.button('Make Predictions'):
        # Loading the TF-IDF vectorizer
        try:
            with open('tfidf_vectorizer.pkl', 'rb') as file:
                tfidf_vectorizer = pickle.load(file)
        except Exception as e:
            st.error(f"Error occurred while loading TF-IDF vectorizer: {e}")
            tfidf_vectorizer = None
        
        # Making predictions
        if tfidf_vectorizer is not None and all((loaded_model_a, loaded_model_b, loaded_model_c)):
            predictions = make_predictions(preprocessed_tweet, tfidf_vectorizer, loaded_model_a, loaded_model_b, loaded_model_c)
            
            # Display predictions
            if predictions:
                st.subheader('Predictions:')
                prediction_str = ' '.join(predictions)
                st.write(prediction_str)
if __name__ == '__main__':
    main()
