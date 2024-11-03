import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Loading a Pretrained Model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Function to search for relevant documents
def search(query, documents, top_k=40):
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

    # Find the most relevant documents
    scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    top_results = scores.topk(top_k)

    # Collect results into a list
    return [(documents[idx], labels[idx], score.item()) for score, idx in zip(top_results.values, top_results.indices)]

# Function for calculating quality metrics
def calculate_metrics(relevant_labels, retrieved_docs, labels):
    # Get labels for found documents
    retrieved_labels = [labels[documents.index(doc)] for doc in retrieved_docs if doc in documents]

    # True Positives (TP)
    true_positives = sum(1 for label in retrieved_labels if label in relevant_labels)

    # False Positives (FP)
    false_positives = len(retrieved_labels) - true_positives

    # Precision
    precision = true_positives / len(retrieved_docs) if retrieved_docs else 0

    return precision

# Streamlit UI
st.title("Document Information Search")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with documents", type="csv")

if uploaded_file is not None:
    # Reading CSV
    df = pd.read_csv(uploaded_file)

    # Checking for the existence of 'text' and 'labels' columns
    if 'text' not in df.columns or 'labels' not in df.columns:
        st.error("The file must contain columns named 'text' and 'labels'.")
    else:
        documents = df['text'].tolist()
        labels = df['labels'].tolist()

        # Query input
        query = st.text_input("Enter your query:")

        if query:
            # Searching
            with st.spinner("Searching..."):
                results = search(query, documents)

            # Displaying results
            st.write("Search Results:")
            retrieved_docs = []
            for doc, label, score in results:
                st.write(f"**Relevance**: {score:.2f} | **Category**: {label}")
                st.write(doc)
                st.write("---")
                retrieved_docs.append(doc)

            # Determine relevant labels for the query
            relevant_labels = {label for doc, label in zip(documents, labels) if label in query}

            # Calculating quality metrics
            precision = calculate_metrics(relevant_labels, retrieved_docs, labels)
            st.write(f"**Precision@{len(retrieved_docs)}**: {precision:.2f}")
