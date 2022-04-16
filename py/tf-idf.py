## Exploring song lyric data
## Layla Bouzoubaa
## 14/04/2022
## this script calculates TF-IDF scores for each word in each song
###############################################################################

# load packages
import pandas as pd
import numpy as np

# read data
lyrics = pd.read_csv('detect-drug-lm/data/lyrics/lyrics-data.csv')

# center data
# english songs only
lyrics_en = lyrics[lyrics['language'] == 'en']
# keep only song and lyric columns and rename
lyrics_en = lyrics_en[['SName', 'Lyric']].rename(columns={'SName': 'song', 'Lyric': 'lyrics'})
# get lyric count
lyrics_en['count'] = lyrics_en['lyrics'].str.split().str.len()
# remove songs with count < 50 and > 500
lyrics_clean = lyrics_en[lyrics_en['count'] >= 50]
lyrics_clean = lyrics_en[lyrics_en['count'] <= 500]

# remove uneccessary characters
lyrics_clean['lyrics'] = lyrics_clean['lyrics'].str.replace('[^a-zA-Z]', '', regex=True).str.lower()

# NLP EDA --------------------------------------------------------------------

# tokenize lyrics
lyrics_clean['lyrics'] = lyrics_clean['lyrics'].str.split()


# remove stop words
import requests
stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stopwords = set(stopwords_list.decode().splitlines()) 
# extra stop words
extra_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
                         'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
                         "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
                         'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                         'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                         'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
                         'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                         'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                         'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                         'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                         'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
                         'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
                         'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                         "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                         "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                         'won', "won't", 'wouldn', "wouldn't", 'dont', 'didnt', 'doesnt', 'cause', 'cant', 'couldnt', 'im',
                         'youre', 'oh', 'wont', 'ill', 'ive', 'hes', 'thats', 'whos', 'whats', 'youd', 'youve', 'youll',
                   'shouldve', 'chorus', 'lyrics', 'ooh', 'oooh', 'aint', 'verse', 'nah', 'ooo', 'ohoh', 'mm', 'woa', 'woah', 'woo',
                   'huh', 'hmm', 'ohh', 'mm', 'uh', 'whoa', 'cuz', 'ya']
full_stopwords = stopwords.union(extra_stopwords)
lyrics_clean['lyrics'] = lyrics_clean['lyrics'].apply(lambda x: [item for item in x if item not in full_stopwords])

# tf-idf ----------------------------------------------------------------------

# stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=10000)
# tfidf expects array of strings not list of strings
lyrics_clean['lyrics'] = lyrics_clean['lyrics'].apply(lambda x: ' '.join(x))
tfidf_matrix = tfidf.fit_transform(lyrics_clean['lyrics'])


