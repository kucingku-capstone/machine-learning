from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)


model = tf.keras.models.load_model('E:\machine-learning\model\model elo\CF_model_kucingku.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_id = np.array([data['user_id']])
    cat_id = np.array([data['cat_id']])
    gender = np.array([data['gender']])
    size = np.array([data['size']])
    age = np.array([data['age']])

    cat_indices = np.arange(model.layers[6].input_shape[1])  # atau sesuaikan indeks jika diperlukan

    user_input = np.repeat(np.array([user_id]), len(cat_indices))
    additional_features = np.array([gender, age, size])

    input_data = [user_input, additional_features, cat_indices]

    predictions = model.predict(input_data)
    top_cat_indices = np.argsort(predictions.flatten())[::-1][:5]

    # Ubah ini jika Anda menggunakan encoder lain atau ingin format yang berbeda
    top_cat_ids = top_cat_indices.tolist()

    return jsonify(top_cat_ids)

if __name__ == '__main__':
    app.run(debug=True)