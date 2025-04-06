import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from utils import load_data, preprocess_data, recommend_assessments

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Check available models
available_models = [model.name for model in genai.list_models()]
print("Available Models:", available_models)

# Use the correct model name
model_name = "models/gemini-1.5-pro-latest"  # Replace with the correct model name
if model_name not in available_models:
    raise ValueError(f"Model '{model_name}' is not available. Available models: {available_models}")

model = genai.GenerativeModel(model_name)

# Function to extract features using Gemini API
def extract_features(query):
    prompt = f"Extract key features from the following hiring query: {query}. Focus on skills, duration, and test type."
    response = model.generate_content(prompt)
    return response.text

# Load and preprocess data
assessments = load_data('data.json')
vectorizer, tfidf_matrix = preprocess_data(assessments)

# Streamlit app
st.title("SHL Assessment Recommendation System")

# Input query
query = st.text_area("Enter your query or job description:")

if st.button("Get Recommendations"):
    if query:
        # Extract features from the query
        features = extract_features(query)
        
        # Get recommendations
        recommendations = recommend_assessments(features, vectorizer, tfidf_matrix, assessments)
        
        # Display results in a table
        st.table(recommendations)
    else:
        st.warning("Please enter a query.")