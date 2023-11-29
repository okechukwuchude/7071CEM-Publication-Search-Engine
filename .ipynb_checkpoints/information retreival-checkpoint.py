#!/usr/bin/env python
# coding: utf-8

# In[6]:


#mport libraries
import json
import math
import os
import time
import requests
from bs4 import BeautifulSoup
import nltk
from nltk. tokenize import RegexpTokenizer
import schedule
import urllib.robotparser
import tkinter as tk
from tkinter import scrolledtext
import webbrowser


# In[7]:


# Define Stopwords
import requests

# Download the file
url = "https://gist.githubusercontent.com/ZohebAbai/513218c3468130eacff6481f424e4e64/raw/b70776f341a148293ff277afa0d0302c8c38f7e2/gist_stopwords.txt"
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Open the file and write the content
    with open("gist_stopwords.txt", "w") as gist_file:
        gist_file.write(response.text)

        # Read the content, split using commas, and remove double quotes and leading/trailing whitespaces
        stopwords = [i.replace('"', "").strip() for i in response.text.split(",")]

else:
    print(f"Failed to download the file. Status code: {response.status_code}")


# In[8]:


def preprocess_text(text):
    text = text.lower() # Convert the text to lowercase
    processed_text = ''.join([char for char in text if char.isalpha() or char.isspace()]) # Remove punctuation, numbers, and symbols
    tokens = tokenize_text(processed_text) # Tokenize the text
    filtered_tokens = remove_stop_words(tokens) # Remove stop words from the tokens
    stemmed_tokens = stem_words(filtered_tokens) # Stem the tokens
    return stemmed_tokens

def tokenize_text(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    return tokenizer.tokenize(text)

def remove_stop_words(tokens):
    stop_words = set(stopwords)
    return [token for token in tokens if token not in stop_words]

def stem_words(tokens):
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


# In[9]:


#inverted Index
def create_inverted_index(documents):

    inverted_index = {} # Inverted index dictionary

    document_id = 0 # Unique document ID for each record

    for document in documents:
        for record in document:
            document_id += 1

            preprocessed_title = preprocess_text(record['title']) # Preprocess the crawled data
            preprocessed_journal = preprocess_text(record['journal'])

            # Build inverted index
            for token in preprocessed_title + preprocessed_journal:
                if token in inverted_index:
                    inverted_index[token].append(document_id)
                else:
                    inverted_index[token] = [document_id]

    return inverted_index


# In[10]:


#Term Frequency - Inverse Document Frequency
def calculate_tfidf(documents, inverted_index):

    # TF-IDF (Term Frequency - Inverse Document Frequency) dictionary
    tfidf_scores = {}

    # Unique document ID for each record
    document_id = 0

    for document in documents:
        for record in document:
            document_id += 1

            # Preprocess the data
            preprocessed_title = preprocess_text(record['title'])
            preprocessed_journal = preprocess_text(record['journal'])

            # Get the required tokens
            document_tokens = preprocessed_title + preprocessed_journal

            # Find out max word count in a record
            max_frequency = max(document_tokens.count(w) for w in document_tokens)

            # For each record, we store the tfidf for each term in another dict
            tfidf_scores[document_id] = {}

            for token in document_tokens:
                tf = document_tokens.count(token) / max_frequency
                idf = math.log(len(documents) / len(inverted_index[token]))
                tfidf_scores[document_id][token] = tf * idf

    return tfidf_scores


# In[11]:


def rank_documents(query, documents, inverted_index, tfidf_scores):

    # Preprocess user query
    query_tokens = preprocess_text(query)

    # Compute the score based on tfidf
    scores = {}
    for token in query_tokens:
        if token in inverted_index:
            for document_id in inverted_index[token]:
                if document_id in scores:
                    scores[document_id] += tfidf_scores[document_id][token]
                else:
                    scores[document_id] = tfidf_scores[document_id][token]

    # Return the sorted (descending) list of scores
    return sorted(scores, key=scores.get, reverse=True)

def can_fetch_url(url, user_agent='*'):
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(urllib.parse.urljoin(url, '/robots.txt'))
    rp.read()
    return rp.can_fetch(user_agent, url)


# In[12]:


def crawl_publications(url):
    response = requests.get(url)

    # Respect robots.txt and avoid overloading the server
    time.sleep(5)

    soup = BeautifulSoup(response.text, "html.parser") # Parse data

    # Extract all publication-related data
    publication_containers = soup.find_all("div", class_="result-container")

    publications = []
    for container in publication_containers:

        # Get title and publication link
        publication_title = container.find("h3", class_="title").text
        publication_link = container.find("a", class_="link")["href"]

        # Get authors and their profiles
        author_elements = container.find_all("a", class_="link person")
        authors = [author.text.strip() for author in author_elements]

        author_profile_elements = container.find_all("a", class_="link person", rel="Person")
        author_profiles = [author["href"] for author in author_profile_elements]

        # Extract date, journal, volume, number of pages, and article ID
        publication_date = container.find("span", class_="date").text
        journal_element = container.find("span", class_="journal")
        publication_journal = journal_element.text.strip() if journal_element else ""

        volume_element = container.find("span", class_="volume")
        publication_volume = volume_element.text.strip() if volume_element else ""

        number_of_pages_element = container.find("span", class_="numberofpages")
        number_of_pages = number_of_pages_element.text.strip() if number_of_pages_element else ""

        article_id = container.find("p", class_="type").text.split()[-1]

        # Construct a dictionary with all extracted information
        publication_data = {
            "title": publication_title,
            "publication_link": publication_link,
            "authors": authors,
            "author_profiles": author_profiles,
            "date": publication_date,
            "journal": publication_journal,
            "volume": publication_volume,
            "number_of_pages": number_of_pages,
            "article_id": article_id,
        }

        publications.append(publication_data)

    return publications


# In[13]:


def get_document_by_id(document_id, documents):
    current_document_id = 0
    for document in documents:
        for record in document:
            current_document_id += 1
            if current_document_id == document_id:
                return record
    return None

def search_engine_gui(documents, inverted_index, tfidf_scores):
    urls = {}  # Dictionary to store URLs associated with tags

    # Set up the GUI
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

        result_text.delete(1.0, tk.END)  # Clear the result text box
        for document_id in ranked_documents:
            document_info = get_document_by_id(document_id, documents)
            if document_info:
                result_text.insert(tk.END, f"Document ID: {document_id}\nTitle: {document_info['title']}\n\n")
                link = document_info['publication_link']
                result_text.insert(tk.END, f"Publication Link: {link}\n\n", f"link{document_id}")  # Tag the link
                urls[f"link{document_id}"] = link  # Add the URL to the urls dictionary
                result_text.tag_bind(f"link{document_id}", "<Button-1>", open_link)  # Bind the click event to the tag
                result_text.tag_config(f"link{document_id}", foreground="blue", underline=1)  # Style the link

        # Display authors and their profile links
            for i, (author, author_profile) in enumerate(zip(document_info['authors'], document_info['author_profiles'])):
                author_link_tag = f"author_link{document_id}_{i}"
                result_text.insert(tk.END, f"Author: {author}, Profile Link: ", "author_label")
                result_text.insert(tk.END, f"{author_profile}\n", author_link_tag)
                urls[author_link_tag] = author_profile
                result_text.tag_bind(author_link_tag, "<Button-1>", open_link)
                result_text.tag_config(author_link_tag, foreground="green", underline=1)

            result_text.insert(tk.END, "\n")

    # Create the label, entry, button, and text widgets
    label = tk.Label(root, text="Enter your query:")
    label.pack()

    entry = tk.Entry(root, width=50)
    entry.pack()

    search_button = tk.Button(root, text="Search", command=search)
    search_button.pack()

    result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=50)

    root.mainloop()



# In[14]:


def crawl_publications_and_save(json_filename):
    publications = []
    for page_number in range(10):
        url = f"https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-fluid-and-complex-systems-fcs/publications/?page={page_number}"
        publications_from_page = crawl_publications(url)
        publications.extend(publications_from_page)

    with open(json_filename, "w") as json_file:
        json.dump(publications, json_file)

    return publications


# In[15]:


def main():

    # This file will store all our crawled publication data
    # This is to avoid multiple crawlings every time the code is run
    json_filename = "publications.json"

    # Check if the JSON file exists or not
    if os.path.exists(json_filename) and os.path.isfile(json_filename):

        # If it exists, then just open the file and load all data into the publications list
        with open(json_filename, "r") as json_file:
            publications = json.load(json_file)
    else:
        # Otherwise, perform web crawling and populate publications
        # Crawl 10 web pages to get all publication information
        publications = crawl_publications_and_save(json_filename)

    # The publications list contains records in the following format:
    # publications[i][j] is a dictionary containing data of the j-th publication present on the i-th page
    # For example, publications[0][0] returns the first record present on the first page

    # Call the relevant functions to create an index and calculate TF-IDF
    index = create_inverted_index(publications)
    tfidf_scores = calculate_tfidf(publications, index)

    # Prompt the user for a query
    query = input("Enter your query: ")

    # Display only the top 10 most relevant search results
    ranked_documents = rank_documents(query, publications, index, tfidf_scores)[:10]

        # Display the relevant information of the retrieved documents
    for document_id in ranked_documents:
        # Find the corresponding document record
        found_record = None
        for publication in publications:
            for record in publication:
                if record['article_id'] == document_id:
                    found_record = record
                    break

        if found_record:
            print("\nDocument ID:", document_id)
            print("Title:", found_record['title'])
            print("Publication link:", found_record['publication_link'])
            print("Authors:", found_record['authors'])
            print("Author's Profile:", found_record['author_profiles'])
            print("Date:", found_record['date'])
            print("Journal:", found_record['journal'])
            print("Volume:", found_record['volume'])
            print("Article ID:", found_record['article_id'])



# In[16]:


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

    # Create an inverted index and calculate TF-IDF
    inverted_index = create_inverted_index(publications)
    tfidf_scores = calculate_tfidf(publications, inverted_index)

    # Run the GUI-based search engine
    search_engine_gui(publications, inverted_index, tfidf_scores)


# In[ ]:




