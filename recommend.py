import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

def load_dataset(filepath: str):
    """Load the dataset and return a DataFrame."""
    df = pd.read_csv(filepath)
    df = df[['Series_Title', 'Overview']].rename(columns={'Series_Title': 'title', 'Overview': 'description'})
    df = df.dropna().reset_index(drop=True)  # Remove rows with missing descriptions
    return df

def custom_tokenizer(text):
    """Extract words only, ignoring numbers and special characters."""
    text = text.lower().strip()  # Normalize text
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    return text.split()  # Simple word splitting

def preprocess_and_vectorize(df):
    """Convert movie descriptions to TF-IDF vectors."""
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=20000,
        ngram_range=(1, 2),  # Capture both single words and bigrams
        tokenizer=custom_tokenizer  # Use custom tokenizer
    )
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    return vectorizer, tfidf_matrix

def recommend_movies(user_query, df, vectorizer, tfidf_matrix, top_n=5):
    """Find and return top N recommended movies based on cosine similarity."""
    if not user_query.strip():
        return []
    
    user_tfidf = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    if np.all(similarities == 0):  # If all similarities are zero, return an empty list
        return []
    
    top_indices = similarities.argsort()[::-1][:top_n]
    recommendations = [(df.iloc[idx]['title'], similarities[idx]) for idx in top_indices]
    
    return recommendations

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", category=UserWarning)  # Suppress token_pattern warning
    
    if len(sys.argv) < 2:
        print("Usage: python recommend.py \"Your movie preference description\"")
        sys.exit(1)
    
    user_input = ' '.join(sys.argv[1:])  # Capture full input query
    dataset_path = "/Users/jason/Downloads/re/imdb_top_1000.csv"  # Ensure this dataset exists in the correct path
    
    df = load_dataset(dataset_path)
    vectorizer, tfidf_matrix = preprocess_and_vectorize(df)
    
    # Debugging: Check tokenized query words
    processed_query = custom_tokenizer(user_input)
    print("Processed Query Words:", processed_query)
    
    recommendations = recommend_movies(user_input, df, vectorizer, tfidf_matrix)
    
    if not recommendations:
        print("No relevant recommendations found. Try a different description.")
    else:
        print("Top Recommendations:")
        for title, score in recommendations:
            print(f"{title} (Similarity: {score:.4f})")
