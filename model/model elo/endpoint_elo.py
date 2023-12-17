from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)


model = tf.keras.models.load_model('E:\machine-learning\model\model elo\CF_model_kucingku.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_id = data['user_id']
    gender = data.get('gender', 'default_gender')
    age = data.get('age', 'default_age')
    size = data.get('size', 'default_size')

    cat_indices = np.arange(model.layers[2].input_dim)  # Sesuaikan dengan jumlah kucing

    user_input = np.repeat(np.array([user_id]), len(cat_indices))
    additional_features = np.array([gender, age, size])

    input_data = [user_input, additional_features, cat_indices]

    predictions = model.predict(input_data)
    top_cat_indices = np.argsort(predictions.flatten())[::-1][:5]

    # Ubah ini jika Anda menggunakan encoder lain atau ingin format yang berbeda
    top_cat_ids = top_cat_indices.tolist()

    return jsonify(top_cat_ids)

if __name__ == 'main':
    app.run(debug=True)