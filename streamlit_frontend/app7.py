#importing libraries and dependencies
import streamlit as st
import pickle
import pandas as pd
from utils import load_models, TweetProcessor,  make_predictions

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

    # Loading the trained models and TF-IDF vectorizer
    loaded_model_a, loaded_model_b, loaded_model_c, tfidf_vectorizer = load_models()

    # Button to make predictions
    if st.button('Make Predictions'):
        # Making predictions using the function from utils
        if all((loaded_model_a, loaded_model_b, loaded_model_c)):
            predictions = make_predictions(preprocessed_tweet, tfidf_vectorizer, loaded_model_a, loaded_model_b, loaded_model_c)
            
            # Display predictions
            if predictions:
                st.subheader('Predictions:')
                prediction_str = ' '.join(predictions)
                st.write("The tweet is " ,prediction_str)

if __name__ == '__main__':
    main()
