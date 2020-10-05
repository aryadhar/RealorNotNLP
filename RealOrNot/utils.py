import pandas as pd
import numpy as np
from loguru import logger
import re

# For plotting
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from sklearn import preprocessing, decomposition
import gensim
from gensim.utils import simple_preprocess
import spacy  # for lemmatization

from sklearn.feature_extraction.text import (
    TfidfTransformer,
    CountVectorizer,
    TfidfVectorizer,
)

from wordcloud import WordCloud

nltk.download("stopwords")
stop_words = stopwords.words("english")

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en", disable=["parser", "ner"])

# to make pipelines
from sklearn.pipeline import make_pipeline

def _remove_punctuation_and_symbols(text):
    # search for these symbols and replace with empty string
    text = re.sub(r"[!?@#$+%*:;()/'-.<>]", "", text)
    #cleanr = re.compile("<.*?>")
    #cleaned_text = re.sub(cleanr, "", text)
    return text

def _remove_stopwords(text):
    #lower case
    text=text.lower()

    #remove digits
    text=re.sub("[0-9]+", "", text)

    #remove stopwords
    tokenizer= RegexpTokenizer(r"\w+")
    tokens=tokenizer.tokenize(text)
    filtered_words=[w for w in tokens if len(w) > 2 if w not in stop_words]
    return " ".join(filtered_words)

#to applz count encoding and TF-IDF
def apply_tfidf_count(text):
    
    count_vect = CountVectorizer()
    text_count = count_vect.fit_transform(text)
    
    model=make_pipeline(CountVectorizer(), TfidfTransformer())
    resulting_text=model.fit_transform(text)
    
    logger.info(resulting_text.shape)
    
    resulting_df=pd.DataFrame(resulting_text.todense(), columns=text_count.get_feature_names())
    
    return resulting_df

#to remove punctuations using Gensim
def sent_to_words(sentences):
    for sentence in sentences:
        yield (
            simple_preprocess(str(sentence), deacc=True)
        )  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts
    ]

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out

