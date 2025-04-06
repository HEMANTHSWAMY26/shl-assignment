from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai
from utils import load_data, preprocess_data, recommend_assessments

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# Function to extract features using Gemini API
def extract_features(query):
    prompt = f"Extract key features from the following hiring query: {query}. Focus on skills, duration, and test type."
    response = model.generate_content(prompt)
    return response.text

# Load and preprocess data
assessments = load_data('data.json')
vectorizer, tfidf_matrix = preprocess_data(assessments)

# FastAPI app
app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/recommend/")
async def recommend(query: Query):
    # Extract features from the query
    features = extract_features(query.text)
    
    # Get recommendations
    recommendations = recommend_assessments(features, vectorizer, tfidf_matrix, assessments)
    
    # Return recommendations as JSON
    return recommendations.to_dict(orient='records')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)