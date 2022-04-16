## Exploring song lyric data
## Layla Bouzoubaa
## 14/04/2022
## this script performs topic modeling on lyrics to find potential 
# signal for further anaylysis pertaining to drug references
###############################################################################

# load packages
import pandas as pd
import numpy as np
import nltk

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

# take sample of data
lyrics_sample = lyrics_clean.sample(n=10000)

# remove apostrophes and punctuation
lyrics_sample['lyrics'] = lyrics_sample['lyrics'].str.replace('[^\w\s]', '', regex=True)
# remove uneccessary characters
lyrics_sample['lyrics'] = lyrics_sample['lyrics'].str.replace('[^a-zA-Z]', ' ', regex=True).str.lower()



# LDA part deux

from nltk.tokenize import RegexpTokenizer
lyric_corpus_tokenized = []
tokenizer = RegexpTokenizer(r'\w+')
for lyric in lyrics_sample['lyrics']:
    tokenized_lyric = tokenizer.tokenize(lyric.lower())
    lyric_corpus_tokenized.append(tokenized_lyric)
    
# removing numeric tokens or tokenss with less than 3 characters
for s,song in enumerate(lyric_corpus_tokenized):
    filtered_song = []    
    for token in song:
        if len(token) > 2 and not token.isnumeric():
            filtered_song.append(token)
    lyric_corpus_tokenized[s] = filtered_song
    
# token lemmatization
# RUN THIS TO INITIATE NLTK INSTALLER
# import nltk
# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()

import nltk
# nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
for s,song in enumerate(lyric_corpus_tokenized):
    lemmatized_tokens = []
    for token in song:
        lemmatized_tokens.append(lemmatizer.lemmatize(token))
    lyric_corpus_tokenized[s] = lemmatized_tokens
    
# remove stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
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
stop_words.extend(extra_stopwords)
for s,song in enumerate(lyric_corpus_tokenized):
    filtered_text = []    
    for token in song:
        if token not in stop_words:
            filtered_text.append(token)
    lyric_corpus_tokenized[s] = filtered_text
    
# dictionary creation and occurrence based filtering
from gensim.corpora import Dictionary
dictionary = Dictionary(lyric_corpus_tokenized)
dictionary.filter_extremes(no_below = 100, no_above = 0.8)

# bow and index to dictionary conversion
from gensim.corpora import MmCorpus
gensim_corpus = [dictionary.doc2bow(song) for song in lyric_corpus_tokenized]
temp = dictionary[0]
id2word = dictionary.id2token

# setting model parameters
chunksize = 2000
passes = 20
iterations = 400
num_topics = 6

# model training
from gensim.models import LdaModel
lda_model = LdaModel(
corpus=gensim_corpus,
id2word=id2word,
chunksize=chunksize,
alpha='auto',
eta='auto',
iterations=iterations,
num_topics=num_topics,
passes=passes
)
# calculate coherence score - TAKES A LONG TIME
from gensim.models.coherencemodel import CoherenceModel
coherencemodel = CoherenceModel(model=lda_model, texts=lyric_corpus_tokenized, dictionary=dictionary, coherence='c_v')
print(coherencemodel.get_coherence())

# visualize the LDA with pyLDAvis
import pyLDAvis.gensim
pyLDAvis_data = pyLDAvis.gensim.prepare(lda_model, gensim_corpus, dictionary)
pyLDAvis.display(pyLDAvis_data)
pyLDAvis.save_html(pyLDAvis_data, './Lyrics_LDA_k_'+ str(num_topics) +'.html')