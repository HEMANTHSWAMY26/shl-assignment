import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data from JSON
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    assessments = pd.DataFrame(data['assessments'])
    assessments['remote'] = assessments['remote'].apply(lambda x: "Yes" if x else "No")
    assessments['adaptive'] = assessments['adaptive'].apply(lambda x: "Yes" if x else "No")
    assessments['combined'] = assessments.apply(
        lambda row: f"{row['name']} {row['description']} {row['category']} {row['duration']} {' '.join(row['languages'])}", axis=1
    )
    return assessments

# Preprocess and vectorize data
def preprocess_data(assessments):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(assessments['combined'])
    return vectorizer, tfidf_matrix

# Recommend assessments based on query
def recommend_assessments(query, vectorizer, tfidf_matrix, assessments, top_n=10):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommendations = assessments.iloc[top_indices]
    return recommendations[['name', 'url', 'remote', 'adaptive', 'duration', 'category']]