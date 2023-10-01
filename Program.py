import os
import re
import nltk
import redis
import numpy as np
from nltk.corpus import stopwords
from collections import defaultdict

nltk.download('stopwords')

#this function is to preprocess the text from the documents
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

#this function is to build the inverted index
def build_inverted_index(docs_folder):
    inverted_index = defaultdict(lambda: defaultdict(list))
    tf = defaultdict(lambda: defaultdict(int))
    df = defaultdict(int)

    # Building inverted index and calculating TF and DF
    for doc_id, filename in enumerate(os.listdir(docs_folder)):
        with open(os.path.join(docs_folder, filename), 'r') as file:
            text = file.read()
            preprocessed_words = preprocess(text)
            term_count = len(preprocessed_words)
            for pos, term in enumerate(preprocessed_words):
                inverted_index[term][doc_id].append(pos)
                tf[doc_id][term] += 1

                # Update inverted index for phrases
                for phrase_length in range(2, min(5, term_count - pos) + 1):  # Consider phrases of length 2 to 4
                    phrase = " ".join(preprocessed_words[pos:pos + phrase_length])
                    inverted_index[phrase][doc_id].append(pos)

            for term in set(preprocessed_words):
                df[term] += 1  # Increment df for each term

    # Calculate IDF
    total_docs = len(os.listdir(docs_folder))
    idf = {term: np.log(total_docs / df[term]) for term in df}

    # Calculate TF-IDF
    tfidf = {}
    for doc_id in tf:
        tfidf[doc_id] = {}
        for term in tf[doc_id]:
            tfidf[doc_id][term] = (1 + np.log(tf[doc_id][term])) * idf[term]

    return tfidf, inverted_index, df

#this function is to rank documents and calculate the relevance scores
def rank_documents(query, tfidf, inverted_index):
    query_terms = preprocess(query)
    query_vector = defaultdict(float)
    for term in query_terms:
        query_vector[term] += 1

    # Compute relevance scores
    relevance_scores = defaultdict(float)
    for term in query_terms:
        if term in inverted_index:
            for doc_id in inverted_index[term]:
                relevance_scores[doc_id] += query_vector[term] * tfidf[doc_id][term]

    # Sort and return ranked document IDs
    ranked_document_ids = sorted(relevance_scores.keys(), key=lambda doc_id: relevance_scores[doc_id], reverse=True)
    return ranked_document_ids, relevance_scores

#this function is to do the phrase search
def phrase_search(phrase, inverted_index):
    phrase_terms = preprocess(phrase)
    matching_docs = defaultdict(list)

    # Find documents containing the entire phrase
    phrase_key = " ".join(phrase_terms)
    if phrase_key in inverted_index:
        for doc_id in inverted_index[phrase_key]:
            positions = inverted_index[phrase_key][doc_id]
            matching_docs[doc_id] = positions

    return matching_docs

#this function is to update text inside the document
def update_document(doc_id, updated_text, tfidf, inverted_index):
    # Delete old document's positions from inverted index and TF-IDF
    old_text = open(os.path.join(docs_folder, os.listdir(docs_folder)[doc_id]), 'r').read()
    old_preprocessed_words = preprocess(old_text)
    for term in old_preprocessed_words:
        if doc_id in inverted_index[term]:
            inverted_index[term][doc_id].clear()
        if doc_id in tfidf and term in tfidf[doc_id]:
            del tfidf[doc_id][term]

    # Update the document with new text
    with open(os.path.join(docs_folder, os.listdir(docs_folder)[doc_id]), 'w') as file:
        file.write(updated_text)

    # Update inverted index and TF-IDF for the new document
    new_text = open(os.path.join(docs_folder, os.listdir(docs_folder)[doc_id]), 'r').read()
    new_preprocessed_words = preprocess(new_text)
    term_count = len(new_preprocessed_words)
    for pos, term in enumerate(new_preprocessed_words):
        inverted_index[term][doc_id].append(pos)
    tfidf[doc_id] = {}
    for term in set(new_preprocessed_words):
        tfidf[doc_id][term] = (1 + np.log(new_preprocessed_words.count(term))) * idf[term]

#this function is to delete a document
def delete_document(doc_id, tfidf, inverted_index):
    # Delete document's terms from inverted index and TF-IDF
    old_text = open(os.path.join(docs_folder, os.listdir(docs_folder)[doc_id]), 'r').read()
    old_preprocessed_words = preprocess(old_text)
    for term in old_preprocessed_words:
        if doc_id in inverted_index[term]:
            inverted_index[term][doc_id].clear()
        if doc_id in tfidf:
            del tfidf[doc_id][term]

    # Remove the document
    os.remove(os.path.join(docs_folder, os.listdir(docs_folder)[doc_id]))

docs_folder = "docs"  # Folder containing .txt files
tfidf_scores, inverted_index, df = build_inverted_index(docs_folder)
total_docs = len(os.listdir(docs_folder))
idf = {term: np.log(total_docs / df[term]) for term in df}

# Initialize external key-value store (Redis)
r = redis.StrictRedis(host='localhost', port=6379, db=0)

while True:
    print("\n1. Regular Query (TF-IDF Ranking)")
    print("2. Phrase Search")
    print("3. Update Document")
    print("4. Delete Document")
    print("5. Exit")
    choice = int(input("Enter your choice: "))
    
    if choice == 1:
        query = input("Enter your query: ")
        result_document_ids, relevance_scores = rank_documents(query, tfidf_scores, inverted_index)
        print("Ranked Documents:")
        for doc_id in result_document_ids:
            doc_name = os.listdir(docs_folder)[doc_id]
            print(f"Document Name: {doc_name}, Relevance Score: {relevance_scores[doc_id]}")
    
    elif choice == 2:
        phrase = input("Enter the phrase: ")
        matching_docs = phrase_search(phrase, inverted_index)
        if not matching_docs:
            print("No documents matching the phrase.")
        else:
            print("Documents matching the phrase:")
            for doc_id, positions in matching_docs.items():
                doc_name = os.listdir(docs_folder)[doc_id]
                print(f"Document Name: {doc_name}, Positions: {positions}")
    
    elif choice == 3:
        doc_id = int(input("Enter the document ID to update: "))
        updated_text = input("Enter the updated text: ")
        update_document(doc_id, updated_text, tfidf_scores, inverted_index)
        print("Document updated successfully.")
    
    elif choice == 4:
        doc_id = int(input("Enter the document ID to delete: "))
        delete_document(doc_id, tfidf_scores, inverted_index)
        print("Document deleted successfully.")
    
    elif choice == 5:
        break
