import pandas as pd
import numpy as np
import spacy
import string
import re
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError
import uvicorn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import pickle
import logging
from pymongo import MongoClient
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.secret_key = 'PASS@152bn'
nlp = spacy.load("en_core_web_sm")
label_encoder = LabelEncoder()
v = TfidfVectorizer()

# Load the saved model
try:
    with open('recommender.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as err:
    logging.error(f"Unexpected {err=}, {type(err)=}")
    model = None

# Connect to MongoDB
client = MongoClient('mongodb+srv://mahmoudrdwan32:123123123@elmentor.fphwgku.mongodb.net/?retryWrites=true&w=majority')
db = client['elmentor']
collection = db['users']

# Function to retrieve data from MongoDB and store it in the session
def fetch_and_backup_data():
    try:
       global backup_data
       cursor = collection.find({'mentor': True})
       data_list = list(cursor)
       for item in data_list:
           item['_id'] = str(item['_id'])
       backup_data = pd.DataFrame(data_list)
       return backup_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from MongoDB: {e}")

# Preprocessing function
stop_words = spacy.lang.en.stop_words.STOP_WORDS
def preprocess(text):
    if isinstance(text, list):  # Check if text is a list
        text = ' '.join(text)  # Convert list to string
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_alpha and token.text.lower() not in stop_words:
            token_text = token.text.lower()
            token_text = token_text.translate(str.maketrans('', '', string.punctuation))
            lemmatized_token = token.lemma_
            filtered_tokens.append(lemmatized_token)
    return " ".join(filtered_tokens)

# Define Pydantic model for input validation
class MentorRequest(BaseModel):
    _id: str

def get_sorted_neighbor_indexes(distances, indices):
    flattened_distances = distances.flatten()
    flattened_indices = indices.flatten()
    distance_index_pairs = list(zip(flattened_distances, flattened_indices))
    sorted_distance_index_pairs = sorted(distance_index_pairs, key=lambda x: x[0])
    sorted_neighbor_indexes = [int(pair[1]) for pair in sorted_distance_index_pairs]
    return sorted_neighbor_indexes

def replace_nan_with_none(selected_rows_dict):
    for row in selected_rows_dict:
        for key, value in row.items():
            if isinstance(value, list):
                # Handle lists or arrays
                row[key] = [None if (isinstance(v, float) and np.isnan(v)) else v for v in value]
            elif isinstance(value, float) and np.isnan(value):
                # Handle individual float values
                row[key] = None
    return selected_rows_dict

# Route to recommend
@app.post("/recommend")
async def predict(mentor_request: MentorRequest):
    
    # Receive new data
    backup_data = fetch_and_backup_data()
    data_df = backup_data.copy()
    """try:
        # Convert input data to DataFrame
        data_dicts = [user.dict() for user in data]
        data_df = pd.DataFrame(data_dicts)
    except ValidationError as e:
        logging.error(f"Data validation error: {e}")
        raise HTTPException(status_code=422, detail="Invalid data format")"""

    try:
        professional_titles = data_df['professionalTitle'].astype(str)
        specializations = data_df['specialization'].astype(str)
        tech_stacks = data_df['techStack'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    except KeyError as e:
        logging.error(f"Missing key in the JSON data: {e}")
        raise HTTPException(status_code=400, detail=f"Missing key: {e}")

    try:
        data_df['preprocessed_data'] = professional_titles + ' ' + specializations + ' ' + tech_stacks
        data_df['preprocessed_data'] = data_df['preprocessed_data'].apply(preprocess)
        data_df['preprocessed_data'] = data_df['preprocessed_data'].str.replace('\n', ' ').str.replace('||', ' ').str.replace(',', ' ').str.replace('  ', ' ').str.replace(':', ' ')
        data_df['levelOfExperience'] = label_encoder.fit_transform(data_df['levelOfExperience'])
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")
    
    try:
        # Vectorize preprocessed data
        vectorized_data = v.fit_transform(data_df['preprocessed_data'])
        tfidf_df = pd.DataFrame(vectorized_data.toarray(), columns=v.get_feature_names_out())
        tfidf_df.fillna(0, inplace=True)
        df_processed = pd.concat([data_df[['levelOfExperience', 'userName']], tfidf_df], axis=1)
        df_processed.set_index('userName', inplace=True)
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")
    

    # Fitting data into model
    df_processed_matrix = csr_matrix(df_processed.values)
    model.fit(df_processed_matrix)

    mentor_id = mentor_request._id
    mentor_data = df_processed[df_processed['_id'] == mentor_id]

    data_point_matrix = csr_matrix(mentor_data.values.reshape(1, -1))

    distances, indices = model.kneighbors(data_point_matrix, n_neighbors=4)

    distances, indices = model.kneighbors(data_point_matrix, n_neighbors=4)
    sorted_indexes = get_sorted_neighbor_indexes(distances, indices)

    selected_rows = backup_data.iloc[sorted_indexes]
    selected_rows_dict = selected_rows.to_dict(orient="records")

    selected_rows_dict = replace_nan_with_none(selected_rows_dict)

    return {"recommended mentors ": selected_rows_dict}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
