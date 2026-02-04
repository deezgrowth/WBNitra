#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:13:56 2026

@author: dtk
"""

import streamlit as st
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PART 1: THE BRAINS (Same logic as before) ---

@st.cache_resource
def load_and_train_model():
    """
    Loads JSON and trains the model. 
    Cached by Streamlit so it only runs once.
    """
    # 1. Load Data
    try:
        with open('faq_data.json', 'r') as file:
            data = json.load(file)
        faq_list = data['questions']
    except FileNotFoundError:
        return None, None, None

    # 2. Train Model
    questions = [item['question'] for item in faq_list]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(questions)
    
    return faq_list, vectorizer, tfidf_matrix

def get_response(user_input, faq_list, vectorizer, tfidf_matrix):
    user_vec = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)
    best_match_index = np.argmax(similarity_scores)
    confidence_score = similarity_scores[0][best_match_index]

    # Threshold
    if confidence_score < 0.2:
        return "I'm sorry, I don't have an answer for that. Please contact support."
    
    return faq_list[best_match_index]['answer']

# --- PART 2: THE WEB INTERFACE ---

st.title("🤖 Customer Support Bot")
st.markdown("Ask me about hours, refunds, or shipping!")

# Load the brain
faq_list, vectorizer, tfidf_matrix = load_and_train_model()

if faq_list is None:
    st.error("Error: 'faq_data.json' not found. Please create the file in the same directory.")
    st.stop()

# Initialize chat history in the session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Generate Bot Response
    response = get_response(prompt, faq_list, vectorizer, tfidf_matrix)

    # 4. Display Bot message
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # 5. Add bot message to history
    st.session_state.messages.append({"role": "assistant", "content": response})