from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import random

app = Flask(__name__)

# Load your collaborative filtering model and encoders
model = load_model("model/h5/Kucingku_model.h5")
encoders = np.load('cat_encoder.npy', allow_pickle=True).item()

# Load your data (replace with actual data loading code)
df = pd.read_csv("dataset/KucingKu_dataset.csv")

# Recommendation function
def collaborative_filtering_recommendation(model, encoders, user_gender, user_age, cat_gender, cat_age, cat_size, cat_breed, top_n=5, similarity_threshold=0.5):
    # Load encoders
    user_gender_encoder = encoders['user_gender_encoder']
    user_age_encoder = encoders['user_age_encoder']
    cat_gender_encoder = encoders['cat_gender_encoder']
    cat_size_encoder = encoders['cat_size_encoder']
    cat_age_encoder = encoders['cat_age_encoder']
    cat_breed_encoder = encoders['cat_breed_encoder']

    # Encode input features
    try:
        user_gender_encoded = user_gender_encoder.transform([user_gender])[0]
        user_age_encoded = user_age_encoder.transform([user_age])[0]
        cat_gender_encoded = cat_gender_encoder.transform([cat_gender])[0]
        cat_size_encoded = cat_size_encoder.transform([cat_size])[0]
        cat_age_encoded = cat_age_encoder.transform([cat_age])[0]
        cat_breed_encoded = cat_breed_encoder.transform([cat_breed])[0]
    except ValueError as e:
        print(f"error: {e}")
        # Handle unseen labels (e.g., assign a default value or skip the data point)
        return None

    # Make predictions for the user's preferences
    user_preferences = model.predict([
        np.array([user_gender_encoded]),
        np.array([user_age_encoded]),
        np.array([cat_gender_encoded]),
        np.array([cat_size_encoded]),
        np.array([cat_age_encoded]),
        np.array([cat_breed_encoded])
    ])

    # Find users with predicted ratings close to the user's rating
    similar_users = df.loc[
        (df['cat_rating'] >= user_preferences[0][0] - similarity_threshold) & (df['cat_rating'] <= user_preferences[0][0] + similarity_threshold)
    ]

    # Sort similar users by rating and select the top N users
    top_users = similar_users.sample(min(top_n, len(similar_users)))

    # Extract recommended cat IDs from the top users
    recommended_cat_ids = top_users['cat_id'].tolist()

    # Include diverse recommendations
    diverse_recommendations = get_diverse_recommendations()
    print("Diverse Recommendations:", diverse_recommendations)

    # Convert numpy.int64 elements to native Python integers
    recommended_cat_ids = [int(cat_id) for cat_id in recommended_cat_ids]
    diverse_recommendations = [int(cat_id) for cat_id in diverse_recommendations]

    recommended_cat_ids += diverse_recommendations
    print("Final Recommended Cat IDs:", recommended_cat_ids)

    return recommended_cat_ids

def get_diverse_recommendations():
    # Implement logic for diverse recommendations
    # For example, return random cat IDs or popular cat IDs
    # In this example, let's return 3 random cat IDs
    num_diverse_recommendations = 5

    # Ensure diverse recommendations are not duplicates of top recommendations
    diverse_cat_ids = []
    while len(diverse_cat_ids) < num_diverse_recommendations:
        random_cat_id = random.choice(df['cat_id'].unique())
        if random_cat_id not in diverse_cat_ids:
            diverse_cat_ids.append(random_cat_id)

    return diverse_cat_ids

# Recommendation endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_inputs = request.json
        recommended_cats = collaborative_filtering_recommendation(
            model,
            encoders,
            user_inputs["user_gender"],
            user_inputs["user_age"],
            user_inputs["cat_gender"],
            user_inputs["cat_age"],
            user_inputs["cat_size"],
            user_inputs["cat_breed"]
        )
        return jsonify({"recommended_cats_ids": recommended_cats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
