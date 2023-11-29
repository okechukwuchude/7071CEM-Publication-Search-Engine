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

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

########### Define Stopwords #################
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


##### Text Preprocessing ######################
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


def tokenize_text(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    return tokenizer.tokenize(text)

def remove_stop_words(tokens):
    stop_words = set(stopwords)
    return [token for token in tokens if token not in stop_words]

def stem_words(tokens):
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
    

############ inverted index ###################
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

#######Term Frequency - Inverse Document Frequency ###############
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

####### tfidf  score ranking #######################
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

########### Site crawling ########################
def crawl_publications(url):
    response = requests.get(url)

    # Respect robots.txt and avoid overloading the server
    time.sleep(5)

    soup = BeautifulSoup(response.text, "html.parser") # Parse data

    # Extract all publication-related data
    publication_containers = soup.find_all("div", class_="result-container")

    info = []
    for publication in publication_containers:

        #Get title and publication link
        title = publication.find("h3", class_="title").text
        publication_link = publication.find("a", class_="link")["href"]

        # Get authors and their profiles
        authors = publication.find_all("a", class_="link person")
        authors = [author.text for author in authors]
        author_links = publication.find_all("a", class_="link person", rel="Person")
        author_links = [author["href"] for author in author_links]

        # GEt date, journal, volume, number of pages, and article id
        date = publication.find("span", class_="date").text
        journal = publication.find("span", class_="journal")
        if journal is not None:
            journal = journal.get_text()
        else:
            journal = ""
        volume = publication.find("span", class_="volume")
        if volume is not None:
            volume = volume.get_text()
        else:
            volume = ""
        numberofpages = publication.find("span", class_="numberofpages")
        if numberofpages is not None:
            numberofpages = numberofpages.get_text()
        else:
            numberofpages = ""
        article_id = publication.find("p", class_="type").text.split()[-1]

        # Build a dict storing all info and append it to info list
        publication_info = {
            "title": title,
            "publication_link": publication_link,
            "authors": authors,
            "authors_profiles": author_links,
            "date": date,
            "journal": journal,
            "volume": volume,
            "numberofpages": numberofpages,
            "article_id": article_id
        }
        info.append(publication_info)

    return info

# Getting Document ID
def get_document_by_id(document_id, documents):
    current_document_id = 0
    for document in documents:
        for record in document:
            current_document_id += 1
            if current_document_id == document_id:
                return record
    return None

######### Creating the GUI ###############
def search_engine_gui(documents, inverted_index, tfidf_scores):
    url_mapping = {}  # Dictionary to store URLs associated with tags

    root = tk.Tk()
    root.title("Search Engine")

    def open_url_on_click(event):
        # Get the index of the clicked position in the text widget
        clicked_index = result_text.index("@%d,%d" % (event.x, event.y))

        # Find the tag name associated with the clicked position
        clicked_tag = result_text.tag_names(clicked_index)[0]

        # Get the URL associated with the tag name from a dictionary
        url = url_mapping.get(clicked_tag)

        # Open the URL in a web browser if it exists
        if url:
            webbrowser.open(url)

    # Function to perform the search
    def perform_search():
        query = entry.get()
        ranked_documents = rank_documents(query, documents, inverted_index, tfidf_scores)[:10]
        result_text.delete(1.0, tk.END)

        for doc_id in ranked_documents:
            doc_info = get_document_by_id(doc_id, documents)

            if doc_info:
                result_text.insert(tk.END, f"Document ID: {doc_id}\nTitle: {doc_info['title']}\n")
                
                # Insert publication link with clickable functionality
                link = doc_info['publication_link']
                result_text.insert(tk.END, f"Publication Link: {link}\n\n", f"link{doc_id}")
                urls[f"link{doc_id}"] = link
                result_text.tag_bind(f"link{doc_id}", "<Button-1>", open_url_on_click)
                result_text.tag_config(f"link{doc_id}", foreground="blue", underline=1)

            # Display authors and their profile links
            for i, (author, author_link) in enumerate(zip(doc_info['authors'], doc_info['authors_profiles'])):
                author_link_tag = f"author_link{doc_id}_{i}"
                result_text.insert(tk.END, f"Author: {author}, Profile Link: ", "author_label")
                result_text.insert(tk.END, f"{author_link}\n", author_link_tag)
                urls[author_link_tag] = author_link
                result_text.tag_bind(author_link_tag, "<Button-1>", open_url_on_click)
                result_text.tag_config(author_link_tag, foreground="green", underline=1)

            result_text.insert(tk.END, "\n")

    # Create the main window
    root = tk.Tk()
    root.title("Document Search")

    # Create and pack the label, entry, button, and text widgets
    label = tk.Label(root, text="Enter your query:")
    label.pack()

    entry = tk.Entry(root, width=50)
    entry.pack()

    search_button = tk.Button(root, text="Search", command=perform_search)
    search_button.pack()

    result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=50)
    result_text.pack()

    # Dictionary to store URLs for links
    urls = {}

    # Start the GUI event loop
    root.mainloop()

# Store crawled data in JSON FILE
def crawl_publications_and_save(json_filename):
    publications = []
    for page_number in range(9):
        url = crawl_publications(f"https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-fluid-and-complex-systems-fcs/publications/?page={page_number}")
        publications.append(url)

    with open(json_filename, "w") as json_file:
        json.dump(publications, json_file)

    return publications


def search_and_display_results():
    # File to store crawled publication data
    json_filename = "publications.json"

    # Check if the JSON file exists
    if os.path.exists(json_filename) and os.path.isfile(json_filename):
        # If it exists, load data into the publications list
        with open(json_filename, "r") as json_file:
            publications = json.load(json_file)
    else:
        # Otherwise, perform web crawling and populate publications
        publications = crawl_publications_and_save(json_filename)

    # Create an index and calculate TF-IDF scores
    index = create_inverted_index(publications)
    tfidf_scores = calculate_tfidf(publications, index)

    # Prompt the user for a query
    query = input("Enter your query: ")

    # Display the top 10 most relevant search results
    ranked_documents = rank_documents(query, publications, index, tfidf_scores)[:10]

    # Display relevant information for each retrieved document
    for document_id in ranked_documents:
        # Find the corresponding document record
        found_record = 0
        for publication in publications:
            for record in publication:
                found_record += 1 
                if found_record:
                    print("\nDocument ID: ", document_id)
                    print("Title: ", record['title'])
                    print("Publication link: ", record['publication_link'])
                    print("Author: ", record['authors'])
                    print("Author's Profile: ", record['authors_profiles'])
                    print("Date: ", record['date'])
                    print("Journal: ", record['journal'])
                    print("Volume: ", record['volume'])
                    print("Article ID: ", record['article_id'])

if __name__ == "__main__":
    # Define the JSON filename
    json_filename = "publications.json"

    # Check if the JSON file exists
    if os.path.exists(json_filename) and os.path.isfile(json_filename):
        # Load data from the JSON file
        with open(json_filename, "r") as json_file:
            publications = json.load(json_file)
    else:
        # Crawl and save the data if the file doesn't exist
        publications = crawl_publications_and_save(json_filename)

    # Create an inverted index and calculate TF-IDF scores
    inverted_index = create_inverted_index(publications)
    tfidf_scores = calculate_tfidf(publications, inverted_index)

    # Run the GUI-based search engine
    search_engine_gui(publications, inverted_index, tfidf_scores)
