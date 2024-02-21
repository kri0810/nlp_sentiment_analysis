#importing libraries
import streamlit as st
from textblob import TextBlob
from langdetect import detect


# Function to detect language of the tweet
def detect_language(text):
    try:
        language = detect(text)
    except:
        language = 'unknown'
    return language


# Function to analyze sentiment of the tweet
def analyze_sentiment(tweet, language):
    blob = TextBlob(tweet)
    if language != 'en':
        translated_blob = blob.translate(to='en')
        blob = translated_blob
    sentiment_score = blob.sentiment.polarity
    if sentiment_score < 0:
        return "Offensive"
    else:
        return "Not Offensive"



# Main function
def main():
    st.set_page_config(page_title="Offensive Text Detection on Social Media", page_icon=":speech_balloon:")
    st.title("Offensive Text Detection on Social Media")

    # Sidebar
    st.sidebar.header("Input Options")
    language = st.sidebar.selectbox(
        "Select Language of Tweet:",
        ("English", "French", "German", "Spanish")
    )

    # User input tweet
    tweet = st.text_area("Enter your tweet:")

    if st.button("Analyze"):
        if not tweet:
            st.error("Please enter a tweet.")
        else:
            lang_code = 'en'
            if language == "French":
                lang_code = 'fr'
            elif language == "German":
                lang_code = 'de'
            elif language == "Spanish":
                lang_code = 'es'
            else:
                lang_code = 'en'

            # Analyzing sentiment of the tweet
            result = analyze_sentiment(tweet, lang_code)
            st.write("Tweet: ", tweet)
            st.write("Offensiveness: ", result)


if __name__ == "__main__":
    main()