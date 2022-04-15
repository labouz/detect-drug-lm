## Exploring song lyric data
## Layla Bouzoubaa
## 13/04/2022
## this script reads in data on song lyrics to find potential signal for further 
## anaylysis pertaining to drug references
###############################################################################

# load packages
import pandas as pd
import numpy as np
import pyreadr as pr

# read data
lyrics = pd.read_csv('detect-drug-lm/data/lyrics/lyrics-data.csv')

## EDA ------------------------------------------------------------------------
lyrics.head()
# keep only english lyrics
lyrics_en = lyrics[lyrics['language'] == 'en']
# keep only song name and lyrics
lyrics_en = lyrics_en[['SName', 'Lyric']]
# variables - rename to make it easier to read
lyrics_en = lyrics_en.rename(columns={'SName': 'song_name', 'Lyric': 'lyrics'})
# get lyric count
lyrics_en['count'] = lyrics_en['lyrics'].str.split().str.len()
# look at distribution of lyric counts - histogram
lyrics_en['count'].hist(bins=100)
# there are a lot of songs with 0 lyrics - let's remove them ~191k rows
lyrics_en = lyrics_en[lyrics_en['count'] > 0]

# looks like we need to remove miscellaneous characters from the 
# lyrics (e.g. *, -, etc.) and make them lowercase but keep as dataframe
lyrics_clean = lyrics_en['lyrics'].str.replace('[^a-zA-Z]', ' ', regex=True).str.lower()
type(lyrics_clean) #series
# to dataframe
lyrics_clean = pd.DataFrame(lyrics_clean)

## NLP EDA --------------------------------------------------------------------

# read .RDA object for drug names - DOPE package
drug_names = pr.read_r('detect-drug-lm/data/lookup_df.rda')
print(drug_names.keys())
drug_names = drug_names['lookup_df']
# remove records with unknown class and category as they are highly likely 
# to be chemical names
drug_names = drug_names[drug_names.category != 'Unknown']
# separate categories and synonyms into separate lists
drug_cats = drug_names['category'].tolist()
drug_cats = list(set(drug_cats))
drug_syns = drug_names['synonym'].tolist()

# --------------------
# TEST with MARIJUANA
# see how many lyrics contain marijuana
lyrics_clean['hasDrug'] = lyrics_clean['lyrics'].str.contains('marijuana', case=False)
# proportion of lyrics with marijuana - 232/191582
lyrics_clean['hasDrug'].value_counts()
# --------------------

# check if lyrics contain any values in either drug_cats or drug_syns lists
lyrics_clean['hasDrug'] = lyrics_clean['lyrics'].str.contains('|'.join(drug_cats), case=False) #takes too long

# alternate method
for drug in drug_cats:
    lyrics_clean['hasDrug'] = lyrics_clean['lyrics'].str.contains(drug, case=False)
    if lyrics_clean['hasDrug'].any():
        break
# returns only 25 Trues when we know marijuana alone returns 232 true values- something is wrong


# set hasDrug to 1 if the song contains a drug reference
# for i in range(len(lyrics_clean)):
#     if lyrics_clean.iloc[i, 0] in drug_names.synonym.values:
#         lyrics_clean.iloc[i, 1] = 1