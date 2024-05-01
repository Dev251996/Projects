#!/usr/bin/env python
# coding: utf-8

# #### After hydrating the tweets using the hydrator tool, we have .csv files in the path Tweets/

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# #### Concatenating all the csv files to one file Tweets_omicron

# In[2]:


import glob
import pandas as pd

# Set input and output file names and paths
input_path = "Tweets/"
output_file = "Tweets_omicron.csv"

# Use glob to get a list of all CSV files in the input path
all_files = glob.glob(input_path + "*.csv")

# Read all CSV files into a list of dataframes
dfs = [pd.read_csv(file) for file in all_files]

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)

# Write the combined dataframe to a CSV file
combined_df.to_csv(output_file, index=False)


# #### Load the data to dataframe

# In[3]:


import pandas as pd

# Load the tweets data from a CSV file
tweets_df = pd.read_csv("Tweets_omicron.csv")


# In[4]:


tweets_df.shape


# In[5]:


tweets_df.columns


# #### Text preprocessing

# In[6]:


import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Function to clean the text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('utf-8')
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return " ".join(lemmas)

# Apply clean_text to the dataframe
tweets_df["text_processed"] = tweets_df["text"].apply(clean_text)


# In[7]:


tweets_text = tweets_df[['text_processed', 'text', 'id']] #new dataframe with required columns


# In[8]:


tweets_text.head(30)


# #### Sentiment classification as positive, negative or neutral

# In[9]:


get_ipython().system('pip install TextBlob')

from textblob import TextBlob

# Define a function for getting the sentiment of a text using TextBlob
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "positive"
    elif sentiment < 0:
        return "negative"
    else:
        return "neutral"

# Get the sentiment of the tweets
tweets_text["sentiment"] = tweets_text["text"].apply(get_sentiment)


# In[10]:


tweets_text.head(30)


# #### Visualising using a pie graph the percentage of the sentiment of the Omicron tweets

# In[11]:


import matplotlib.pyplot as plt

# Count the number of positive, negative, and neutral tweets
positive_tweets = tweets_text[tweets_text["sentiment"] == "positive"].shape[0]
negative_tweets = tweets_text[tweets_text["sentiment"] == "negative"].shape[0]
neutral_tweets = tweets_text[tweets_text["sentiment"] == "neutral"].shape[0]

# Create a pie chart
labels = ["Positive", "Negative", "Neutral"]
sizes = [positive_tweets, negative_tweets, neutral_tweets]
colors = ["green", "red", "blue"]
plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%")
plt.axis("equal")
plt.title("Sentiment Analysis Results")

# Show the pie chart
plt.show()


# In[ ]:




