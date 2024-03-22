from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_frontend.utils import load_models, TweetProcessor

app = FastAPI()

class TweetRequest(BaseModel):
    tweet: str


def make_predictions(unseen_tweet, tfidf_vectorizer, loaded_model_a, loaded_model_b, loaded_model_c):
    try:
        # Preprocess the unseen tweet
        tweet_processor = TweetProcessor()
        preprocessed_tweet = tweet_processor.process_tweets(pd.DataFrame({'tweet': [unseen_tweet]}))['stemmed_tweet'].iloc[0]

        # Vectorize the preprocessed tweet using the TF-IDF vectorizer
        X_unseen = tfidf_vectorizer.transform([preprocessed_tweet])

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
        raise HTTPException(status_code=500, detail=f"Error occurred while making predictions: {e}")

@app.post("/predict")
def predict(tweet_request: TweetRequest):
    loaded_model_a, loaded_model_b, loaded_model_c, tfidf_vectorizer = load_models()
    predictions = make_predictions(tweet_request.tweet, tfidf_vectorizer, loaded_model_a, loaded_model_b, loaded_model_c)
    return {"predictions": predictions}
