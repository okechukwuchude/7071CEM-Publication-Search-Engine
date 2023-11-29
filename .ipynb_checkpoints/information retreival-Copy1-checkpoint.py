#!/usr/bin/env python
# coding: utf-8

import json
import math
import os
import time
import webbrowser
import requests
import urllib.parse
import urllib.robotparser
import tkinter as tk
from tkinter import scrolledtext
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
import nltk

# Download stopwords file
stopwords_url = "https://gist.githubusercontent.com/ZohebAbai/513218c3468130eacff6481f424e4e64/raw/b70776f341a148293ff277afa0d0302c8c38f7e2/gist_stopwords.txt"
try:
    response = requests.get(stopwords_url)
    response.raise_for_status()
    with open("gist_stopwords.txt", "w") as gist_file:
        gist_file.write(response.text)
    stopwords = [i.replace('"', "").strip() for i in response.text.split(",")]
except requests.exceptions.RequestException as e:
    print(f"Failed to download stopwords. Error: {e}")
    # You might want to handle this error accordingly.

# Tokenization function
def tokenize_text(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    return tokenizer.tokenize(text)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    # Remove non-ASCII characters
    processed_text = ''.join([char if ord(char) < 128 else ' ' for char in text])
    processed_text = ''.join([char for char in processed_text if char.isalpha() or char.isspace()])
    tokens = tokenize_text(processed_text)
    filtered_tokens = remove_stop_words(tokens)
    stemmed_tokens = stem_words(filtered_tokens)
    return stemmed_tokens

# Function to remove stop words
def remove_stop_words(tokens):
    stop_words = set(stopwords)
    return [token for token in tokens if token not in stop_words]

# Function to stem words
def stem_words(tokens):
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

#################
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        # Remove non-ASCII characters
        processed_text = ''.join([char if ord(char) < 128 else ' ' for char in text])
        processed_text = ''.join([char for char in processed_text if char.isalpha() or char.isspace()])
        tokens = tokenize_text(processed_text)
        filtered_tokens = remove_stop_words(tokens)
        stemmed_tokens = stem_words(filtered_tokens)
        return stemmed_tokens
    else:
        return text
#########################

# Create inverted index
def create_inverted_index(documents):
    inverted_index = {}
    document_id = 0

    for document in documents:
        for record in document:
            for key, value in record.items():
                print(f"{key}: {value}")
                
            document_id += 1
            preprocessed_title = preprocess_text(record['title']) 
            preprocessed_journal = preprocess_text(record['journal'])

            for token in preprocessed_title + preprocessed_journal:
                if token in inverted_index:
                    inverted_index[token].append(document_id)
                else:
                    inverted_index[token] = [document_id]
    
    return inverted_index


# Calculate TF-IDF scores
def calculate_tfidf(documents, inverted_index):
    tfidf_scores = {}
    document_id = 0

    for document in documents:
        for record in document:
            document_id += 1
            preprocessed_title = preprocess_text(record['title'])
            preprocessed_journal = preprocess_text(record['journal'])
            document_tokens = preprocessed_title + preprocessed_journal
            max_frequency = max(document_tokens.count(w) for w in document_tokens)

            tfidf_scores[document_id] = {}

            for token in document_tokens:
                tf = document_tokens.count(token) / max_frequency
                idf = math.log(len(documents) / len(inverted_index[token]))
                tfidf_scores[document_id][token] = tf * idf

    return tfidf_scores

# Rank documents based on user query
def rank_documents(query, documents, inverted_index, tfidf_scores):
    query_tokens = preprocess_text(query)
    scores = {}

    for token in query_tokens:
        if token in inverted_index:
            for document_id in inverted_index[token]:
                if document_id in scores:
                    scores[document_id] += tfidf_scores[document_id][token]
                else:
                    scores[document_id] = tfidf_scores[document_id][token]

    return sorted(scores, key=scores.get, reverse=True)

# Check if a URL can be fetched respecting robots.txt
def can_fetch_url(url, user_agent='*'):
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(urllib.parse.urljoin(url, '/robots.txt'))
    rp.read()
    return rp.can_fetch(user_agent, url)

# Crawl publications from a given URL
def crawl_publications(url):
    response = requests.get(url)
    time.sleep(5)  # Respect robots.txt and avoid overloading the server

    soup = BeautifulSoup(response.text, "html.parser")
    publication_containers = soup.find_all("div", class_="result-container")

    publications = []
    for container in publication_containers:
        publication_title = container.find("h3", class_="title").text
        publication_link = container.find("a", class_="link")["href"]
        authors = [author.text.strip() for author in container.find_all("a", class_="link person")]
        author_profiles = [author["href"] for author in container.find_all("a", class_="link person", rel="Person")]
        publication_date = container.find("span", class_="date").text
        publication_journal = container.find("span", class_="journal").text.strip() if container.find("span", class_="journal") else ""
        volume = container.find("span", class_="volume").text.strip() if container.find("span", class_="volume") else ""
        number_of_pages = container.find("span", class_="numberofpages").text.strip() if container.find("span", class_="numberofpages") else ""
        article_id = container.find("p", class_="type").text.split()[-1]

        publication_data = {
            "title": publication_title,
            "publication_link": publication_link,
            "authors": authors,
            "author_profiles": author_profiles,
            "date": publication_date,
            "journal": publication_journal,
            "volume": volume,
            "number_of_pages": number_of_pages,
            "article_id": article_id,
        }

        publications.append(publication_data)  # Store each publication as a dictionary

    return [publications]  # Wrap the list of dictionaries in a list



# Get document by ID from the documents list
def get_document_by_id(document_id, documents):
    current_document_id = 0
    for document in documents:
        for record in document:
            current_document_id += 1
            if current_document_id == document_id:
                return record
    return None

# GUI-based search engine
def search_engine_gui(documents, inverted_index, tfidf_scores):
    urls = {}  # Dictionary to store URLs associated with tags

    root = tk.Tk()
    root.title("Search Engine")

    def open_link(event):
        index = result_text.index("@%d,%d" % (event.x, event.y))
        tag_name = result_text.tag_names(index)[0]
        url = urls.get(tag_name)
        if url:
            webbrowser.open(url)

    def search():
        query = entry.get()
        ranked_documents = rank_documents(query, documents, inverted_index, tfidf_scores)[:10]

        result_text.delete(1.0, tk.END)
        for document_id in ranked_documents:
            document_info = get_document_by_id(document_id, documents)
            if document_info:
                result_text.insert(tk.END, f"Document ID: {document_id}\nTitle: {document_info['title']}\n\n")
                link = document_info['publication_link']
                result_text.insert(tk.END, f"Publication Link: {link}\n\n", f"link{document_id}")
                urls[f"link{document_id}"] = link
                result_text.tag_bind(f"link{document_id}", "<Button-1>", open_link)
                result_text.tag_config(f"link{document_id}", foreground="blue", underline=1)

                for i, (author, author_profile) in enumerate(zip(document_info['authors'], document_info['author_profiles'])):
                    author_link_tag = f"author_link{document_id}_{i}"
                    result_text.insert(tk.END, f"Author: {author}, Profile Link: ", "author_label")
                    result_text.insert(tk.END, f"{author_profile}\n", author_link_tag)
                    urls[author_link_tag] = author_profile
                    result_text.tag_bind(author_link_tag, "<Button-1>", open_link)
                    result_text.tag_config(author_link_tag, foreground="green", underline=1)

                result_text.insert(tk.END, "\n")

    label = tk.Label(root, text="Enter your query:")
    label.pack()

    entry = tk.Entry(root, width=50)
    entry.pack()

    search_button = tk.Button(root, text="Search", command=search)
    search_button.pack()

    result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=50)
    result_text.pack()

    root.mainloop()

# Crawl publications and save to JSON file
def crawl_publications_and_save(json_filename):
    publications = []
    for page_number in range(10):
        url = f"https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-fluid-and-complex-systems-fcs/publications/?page={page_number}"
        publications_from_page = crawl_publications(url)
        publications.extend(publications_from_page)

    with open(json_filename, "w") as json_file:
        json.dump(publications, json_file)

    return publications


def main():

    # This file will store all our crawled data
    # This is to avoid multiple crawlings everytime code is run
    json_file = "publications.json"

    # Check if the JSON file exists or not
    if os.path.exists(json_file) and os.path.isfile(json_file):

        # If exists, then just open the file and load all data to information list
        with open(json_file, "r") as f:
            publications = json.load(f)
    else:
        # Else, perform web crawling and populate information
        # The 10 web pages are crawled to get all publication info
        publications = crawl_publications_and_save(json_file)

    # Information list contains records in the following format:
    # information[i][j] is a dictionary containing data of j'th publication present in i'th page
    # For example, information[0][0] returns the first record present in first page

    # Call relevant function to create index and calculate tf-idf
    index = create_inverted_index(documents)
    tfidf = calculate_tfidf(documents, inverted_index)

    # For user query, get the ranked list of docs
    query = input("Enter your query: ")

    # Only the first 10 most relevant searches are displayed
    ranked_docs = rank_documents(query, documents, inverted_index, tfidf)[:10]

    # Display relevant information of retrieved docs
    for doc_id in ranked_docs:
        doc_id_comp = 0
        for page in documents:
            for record in page:
                doc_id_comp = doc_id_comp + 1
                if doc_id_comp == doc_id:
                    print("\nDocument ID: ", doc_id)
                    print("Title: ", record['title'])
                    print("Publication link: ", record['publication_link'])
                    print("Author: ", record['authors'])
                    print("Author's Profile: ", record['authors_profiles'])
                    print("Date: ", record['date'])
                    print("Journal: ", record['journal'])
                    print("Volume: ", record['volume'])
                    print("Article ID: ", record['article_id'])


if __name__ == "__main__":
    # Assuming 'publications.json' is defined above
    json_filename = "publications.json"

    # Check if the JSON file exists
    if os.path.exists(json_filename) and os.path.isfile(json_filename):
        # Load data from the JSON file
        with open(json_filename, "r") as json_file:
            publications = json.load(json_file)
    else:
        # Crawl and save the data
        publications = crawl_publications_and_save(json_filename)

    # Print the content of the 'publications' list before calling create_inverted_index
    print("Content of 'publications' before calling create_inverted_index:")
    print(publications)

    # Create an inverted index and calculate TF-IDF
    inverted_index = create_inverted_index(publications)
    tfidf_scores = calculate_tfidf(publications, inverted_index)

    # Run the GUI-based search engine
    search_engine_gui(publications, inverted_index, tfidf_scores)

