import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved model and data
model_file = './medicine.pkl'
data_file = './medicines_data.pkl'

# Load the TF-IDF model
tfidf = joblib.load(model_file)

# Load the dataframe and the vectorized matrix
df, tfidf_matrix = joblib.load(data_file)

def recommend_top_medicines(salt1, salt2='', top_n=5):
    # Check if salt2 is provided; if not, just use salt1
    if salt2.strip():  # if salt2 is not empty
        query = salt1 + ' ' + salt2
    else:
        query = salt1

    # Transform the input salt combination to the same TF-IDF vector space
    query_vec = tfidf.transform([query])

    # Compute cosine similarity between the input vector and all the medicine vectors
    similarity = cosine_similarity(query_vec, tfidf_matrix)

    # Get the indices of the top_n most similar medicines
    similar_indices = similarity.argsort()[0][-top_n:][::-1]  # Sorting in descending order

    # Return the top_n most similar medicine names
    return df[['name', 'manufacturer_name', 'combined_composition']].iloc[similar_indices]

# Example usage with both salts:
recommended_medicines = recommend_top_medicines('Ambroxol', 'Clavulanic Acid (125mg)', top_n=5)
print(f"Top 5 Recommended Medicines (with two salts): \n{recommended_medicines}")

# Example usage with only one salt:
recommended_medicines = recommend_top_medicines('Ambroxol', top_n=5)
print(f"Top 5 Recommended Medicines (with one salt): \n{recommended_medicines}")
