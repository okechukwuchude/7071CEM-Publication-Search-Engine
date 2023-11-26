import json
import math
import os
import time
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import RegexpTokenizer
import tkinter as tk
from tkinter import scrolledtext
import webbrowser
import urllib.robotparser

# Initialize tokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

# Define a list of stopwords
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', ...]  # Same list as before

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    tokens = tokenizer.tokenize(text)
    words = [w for w in tokens if w not in stopwords]
    stemmer = nltk.stem.PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return words

def create_index(information):
    index = {}
    doc_id = 0

    for page in information:
        for record in page:
            doc_id += 1
            title_tokens = preprocess(record['title'])
            journal_tokens = preprocess(record['journal'])

            for token in title_tokens + journal_tokens:
                index.setdefault(token, []).append(doc_id)

    return index

def calculate_tfidf(information, index):
    tfidf = {}
    doc_id = 0

    for page in information:
        for record in page:
            doc_id += 1
            doc_tokens = preprocess(record['title']) + preprocess(record['journal'])
            max_freq = max(doc_tokens.count(w) for w in doc_tokens)
            tfidf[doc_id] = {token: (doc_tokens.count(token) / max_freq) * math.log(len(information) / len(index[token]))
                             for token in doc_tokens if token in index}

    return tfidf

def rank_documents(query, information, index, tfidf):
    query_tokens = preprocess(query)
    scores = {}

    for token in query_tokens:
        if token in index:
            for doc_id in index[token]:
                scores[doc_id] = scores.get(doc_id, 0) + tfidf[doc_id].get(token, 0)

    return sorted(scores, key=scores.get, reverse=True)

def can_fetch(url, user_agent='*'):
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(urllib.parse.urljoin(url, '/robots.txt'))
    rp.read()
    return rp.can_fetch(user_agent, url)

def crawler(url):
    response = requests.get(url)
    time.sleep(5)  # Politeness delay
    soup = BeautifulSoup(response.text, "html.parser")
    publications = soup.find_all("div", class_="result-container")

    info = []
    for publication in publications:
        title = publication.find("h3", class_="title").text
        publication_link = publication.find("a", class_="link")["href"]
        authors = [author.text for author in publication.find_all("a", class_="link person")]
        author_links = [author["href"] for author in publication.find_all("a", class_="link person", rel="Person")]
        # ... (rest of the code remains the same)

    return info

# (Remaining functions and main logic stay the same)

# Example usage:
# json_file = "information.json"
# information = crawl_data(json_file)
# index = create_index(information)
# tfidf = calculate_tfidf(information, index)
# gui_search_engine(information, index, tfidf)
