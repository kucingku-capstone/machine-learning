import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model("E:\\machine-learning\\model\\model elo\\CF_model.h5")

# Load cat encoders
encoders = np.load('E:\\machine-learning\\model\\model elo\\cat_encoder.npy', allow_pickle=True).item()

# Sample recommendation function using the loaded model and encoders
def get_kucing_recommendations(user_data, model):
    # Make predictions using the model
    predictions = model.predict([[user_data['user_id']] + list(encoded_data.values())])

    # Format predictions as cat_info dictionaries
    recommendations = [
        {
            "cat_id": cat_id,
            "breed": user_data['breed'],  # Include the original 'breed' value
            "age": encoders['age'].inverse_transform([en['age']])[0],
            "gender": encoders['gender'].inverse_transform([encoded_data['gender']])[0],
            "size": encoders['size'].inverse_transform([encoded_data['size']])[0]
        }
        for cat_id in predictions[0]
    ]

    return recommendations

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    # Get user data from the request's JSON body
    user_data = request.get_json()

    # Call the recommendation function
    recommendations = get_kucing_recommendations(user_data, model)

    # Return the formatted recommendations as JSON response
    return jsonify(recommendations)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
