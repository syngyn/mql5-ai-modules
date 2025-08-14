from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Dummy model setup
scaler = MinMaxScaler()
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(100, 1)),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam', loss='mse')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get("prices", [])
    if len(data) < 100:
        return jsonify({"error": "Need at least 100 prices"}), 400

    scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    X = scaled[-100:].reshape(1, 100, 1)
    pred = model.predict(X)[0]
    pred_rescaled = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

    return jsonify({
        "hourly": pred_rescaled[:5].tolist(),
        "daily": pred_rescaled[5:].tolist()
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)