from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf

# Correct the model loading path
model = tf.keras.models.load_model("E:\machine-learning\model\model elo\CF_model_kucingku.h5")

encoders = np.load('E:\machine-learning\model\model elo\cat_encoder.npy', allow_pickle=True).item()

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input
    user_id = int(request.json['user_id'])

    num_cats = len(encoders['cat_encoder'].classes_)
    cat_indices = np.arange(num_cats)
    gender_indices = np.zeros_like(cat_indices)
    size_indices = np.zeros_like(cat_indices)
    age_indices = np.zeros_like(cat_indices)

    # Predictions
    predictions = model.predict([np.array([user_id] * num_cats).reshape(-1, 1), cat_indices, gender_indices, size_indices, age_indices])

    # Get top recommendations
    top_cat_indices = np.argsort(predictions.flatten())[::-1][:10]
    top_cat_ids = encoders['cat_encoder'].inverse_transform(top_cat_indices)  # Correct the variable name

    # Return recommendations as JSON response
    recommendations = [{"cat_id": int(cat_id)} for cat_id in top_cat_ids]
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
