## Exploring song lyric data
## Layla Bouzoubaa
## 14/04/2022
## this script performs topic modeling on lyrics to find potential 
# signal for further anaylysis pertaining to drug references
###############################################################################

# load packages
import pandas as pd
import numpy as np
import pyreadr as pr

# read data
lyrics = pd.read_csv('detect-drug-lm/data/lyrics/lyrics-data.csv')

# topic modelling -------------------------------------------------------------

# 1. read in data - only english songs
lyrics_en = lyrics[lyrics['language'] == 'en']

# 2. keep only song name and lyrics
lyrics_en = lyrics_en[['SName', 'Lyric']]

# 3. variables - rename to make it easier to read
lyrics_en = lyrics_en.rename(columns={'SName': 'song_name', 'Lyric': 'lyrics'})

# get lyric count
lyrics_en['count'] = lyrics_en['lyrics'].str.split().str.len()
# look at distribution of lyric counts - histogram
# lyrics_en['count'].hist(bins=100)
# there are a lot of songs with 0 lyrics - let's remove them ~189k rows
# median song len ~ 211 words - set cuttoff to songs with more than 50 but less than 500 words 
lyrics_clean = lyrics_en[lyrics_en['count'] >= 50]
lyrics_clean = lyrics_en[lyrics_en['count'] <= 500]

# 4. remove any punctuation or special characters with regex
lyrics_clean['lyrics'] = lyrics_clean['lyrics'].str.replace('[^\w\s]', '', regex=True)
# to lower
lyrics_clean['lyrics'] = lyrics_clean['lyrics'].str.lower() 
# ~177k songs


# prep data for lda -----------------------------------------------------------
import gensim
from gensim.utils import simple_preprocess
import nltk
# returns error
# nltk.download('stopwords')
# from nltk.corpus import stopwords
import requests
stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stopwords = set(stopwords_list.decode().splitlines()) 

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
                         'youre', 'oh', 'wont', 'ill', 'ive', 'hes', 'thats', 'whos', 'whats', 'youd', 'youve', 'youll',]

all_stopwords = stopwords.union(extra_stopwords)
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in all_stopwords] for doc in texts]

data = lyrics_clean.lyrics.values.tolist()
data_words = list(sent_to_words(data))
data_words = remove_stopwords(data_words)

# 5. make a wordcloud of the lyrics
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# # join the lyrics together
# lyric_str = ' '.join(lyrics_en['lyrics'])
# # create wordcloud object
# wordcloud = WordCloud(background_color='white', max_words=5000, contour_color='black', contour_width=3, width=800, height=400).generate(lyric_str)
# wordcloud.to_image()

# TDF -------------------------------------------------------------------------
import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


## LDA ------------------------------------------------------------------------
from pprint import pprint
# number of topics
num_topics = 50
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
# Print the Keyword in the 50 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# coherence analysis
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
# coherence score: .345

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation as LDA

# count_vectorizer = CountVectorizer(stop_words='english')
# count_dat = count_vectorizer.fit(lyrics_en['lyrics'])

# num_topics = 5
# lda = LDA(n_components=num_topics, learning_method='batch')
# lda_dat = lda.fit_transform(count_dat.transform(lyrics_en['lyrics']))
