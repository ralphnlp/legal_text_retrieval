import os
import json
from tqdm import tqdm
from pyvi import ViTokenizer
import pickle
from rank_bm25 import BM25Okapi
import streamlit as st


def load_legal_corpus(path):

    print('\n[LOADING TEXTS]')
    with open(path, 'r') as file:
        corpus = json.load(file)

    texts = []
    for law in tqdm(corpus):
        articles = law['articles']
        for article in articles:
            texts.append(article['text'])
    return texts


def clean_texts(texts):

    print('\n[CLEANING TEXTS]')
    def cleaning_text(text):
        tokens = ViTokenizer.tokenize(text)
        tokens = [token.lower() for token in tokens.split() if token.isalpha() or '_' in token]
        cleaned_text = ' '.join(tokens)
        #cleaned_text = tokens
        return cleaned_text

    cleaned_texts = []
    for text in tqdm(texts):
        cleaned_text = cleaning_text(text)
        cleaned_texts.append(cleaned_text)
    return cleaned_texts


def load_cleaned_texts(path):
    if os.path.exists(path):
        with open('./cleaned_texts.pkl', 'rb') as file:
            cleaned_texts = pickle.load(file)
        return cleaned_texts
    else:
        print('\n[No exists cleaned text path]')
        return None



legal_corpus_path = './dataset/legal_corpus.json'
cleaned_corpus_path = './cleaned_texts.pkl'

corpus = load_legal_corpus(legal_corpus_path)
cleaned_corpus = load_cleaned_texts(cleaned_corpus_path)
if cleaned_corpus== None:
    cleaned_corpus = clean_texts(corpus)
    with open(cleaned_corpus_path, 'wb') as file:
        pickle.dump(cleaned_corpus, file)

cleaned_corpus = [text.split() for text in cleaned_corpus]
bm25 = BM25Okapi(cleaned_corpus)



if __name__=='__main__':
    
    st.title('legal text retrieval')
    query = st.text_input(label='')
    if query != '':
        cleaned_query = clean_texts([query])[0]
        cleaned_query = cleaned_query.split()
        print(clean_texts)
        relavent_article = bm25.get_top_n(cleaned_query, corpus, n=1)
        st.write(relavent_article)