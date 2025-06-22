from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import DQN

app = Flask(__name__)
CORS(app)

# Load Models
point_model = tf.keras.models.load_model("point_LSTM.keras")
lstm_model = tf.keras.models.load_model("autoencoder_LSTM.keras")
transformer_model = tf.keras.models.load_model("transformer_LSTM.keras")
rl_model = DQN.load("rl_model_sb3.zip")

# ---------- Preprocessing ----------
def preprocess(df):
    sensor_columns = [col for col in df.columns if col not in ['timestamp', 'activityID', 'activity_name']]
    sensor_data = df[sensor_columns].interpolate(method='linear').dropna()

    # Engineered features
    sensor_data['hand_acc_magnitude'] = np.linalg.norm(sensor_data[['IMU_hand_3D_acceleration_1',
                                                                     'IMU_hand_3D_acceleration_2',
                                                                     'IMU_hand_3D_acceleration_3']], axis=1)
    sensor_data['chest_acc_magnitude'] = np.linalg.norm(sensor_data[['IMU_chest_3D_acceleration_1',
                                                                      'IMU_chest_3D_acceleration_2',
                                                                      'IMU_chest_3D_acceleration_3']], axis=1)
    sensor_data['chest_gyro_magnitude'] = np.linalg.norm(sensor_data[['IMU_chest_3D_gyroscope_1',
                                                                       'IMU_chest_3D_gyroscope_2',
                                                                       'IMU_chest_3D_gyroscope_3']], axis=1)
    sensor_data['chest_mag_magnitude'] = np.linalg.norm(sensor_data[['IMU_chest_3D_magnetometer_1',
                                                                      'IMU_chest_3D_magnetometer_2',
                                                                      'IMU_chest_3D_magnetometer_3']], axis=1)
    sensor_data['ankle_acc_magnitude'] = np.linalg.norm(sensor_data[['IMU_ankle_3D_acceleration_1',
                                                                      'IMU_ankle_3D_acceleration_2',
                                                                      'IMU_ankle_3D_acceleration_3']], axis=1)
    sensor_data['ankle_gyro_magnitude'] = np.linalg.norm(sensor_data[['IMU_ankle_3D_gyroscope_1',
                                                                       'IMU_ankle_3D_gyroscope_2',
                                                                       'IMU_ankle_3D_gyroscope_3']], axis=1)
    sensor_data['ankle_mag_magnitude'] = np.linalg.norm(sensor_data[['IMU_ankle_3D_magnetometer_1',
                                                                      'IMU_ankle_3D_magnetometer_2',
                                                                      'IMU_ankle_3D_magnetometer_3']], axis=1)
    sensor_data['hand_acc_magnitude_4_6'] = np.linalg.norm(sensor_data[['IMU_hand_3D_acceleration_4',
                                                                         'IMU_hand_3D_acceleration_5',
                                                                         'IMU_hand_3D_acceleration_6']], axis=1)
    sensor_data['hand_gyro_magnitude'] = np.linalg.norm(sensor_data[['IMU_hand_3D_gyroscope_1',
                                                                      'IMU_hand_3D_gyroscope_2',
                                                                      'IMU_hand_3D_gyroscope_3']], axis=1)
    sensor_data['hand_mag_magnitude'] = np.linalg.norm(sensor_data[['IMU_hand_3D_magnetometer_1',
                                                                     'IMU_hand_3D_magnetometer_2',
                                                                     'IMU_hand_3D_magnetometer_3']], axis=1)
    sensor_data['chest_acc_magnitude_4_6'] = np.linalg.norm(sensor_data[['IMU_chest_3D_acceleration_4',
                                                                          'IMU_chest_3D_acceleration_5',
                                                                          'IMU_chest_3D_acceleration_6']], axis=1)
    sensor_data['ankle_acc_magnitude_4_6'] = np.linalg.norm(sensor_data[['IMU_ankle_3D_acceleration_4',
                                                                          'IMU_ankle_3D_acceleration_5',
                                                                          'IMU_ankle_3D_acceleration_6']], axis=1)

    # Drop raw sensor cols
    drop_cols = [col for col in sensor_data.columns if 'acceleration' in col or 'gyroscope' in col or 'magnetometer' in col]
    sensor_data.drop(columns=drop_cols, inplace=True)

    # Scale data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(sensor_data)
    scaled_df = pd.DataFrame(scaled, columns=sensor_data.columns)

    # Keep timestamp metadata for alignment
    scaled_df['timestamp'] = df['timestamp'].values[:len(scaled_df)]
    scaled_df['activityID'] = df['activityID'].values[:len(scaled_df)]
    scaled_df['activity_name'] = df['activity_name'].values[:len(scaled_df)]

    return scaled_df

def create_sequences(data, length):
    return np.array([data.iloc[i:i+length].drop(columns=['timestamp', 'activityID', 'activity_name']).values
                     for i in range(len(data) - length)])

# ---------- Upload Route ----------
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        df = pd.read_csv(file)
        df_processed = preprocess(df)

        # Short sequences (50 timesteps)
        X_short = create_sequences(df_processed, 50)
        error_point = np.mean(np.abs(point_model.predict(X_short) - X_short), axis=(1, 2))
        error_lstm = np.mean(np.abs(lstm_model.predict(X_short) - X_short), axis=(1, 2))

        # Long sequences (100 timesteps)
        X_long = create_sequences(df_processed, 100)
        error_trans = np.mean(np.abs(transformer_model.predict(X_long) - X_long), axis=(1, 2))

        # Combine contextual reconstruction errors
        min_len = min(len(error_lstm), len(error_trans))
        combined_context = error_lstm[:min_len] + error_trans[:min_len]
        norm_context = (combined_context - np.min(combined_context)) / (np.max(combined_context) - np.min(combined_context))
        X_rl = np.expand_dims(norm_context, axis=-1)

        # RL-based predictions
        rl_preds = np.array([rl_model.predict(obs, deterministic=True)[0] for obs in X_rl])

        # Point anomaly threshold and ratio
        threshold_point = np.percentile(error_point, 95)
        point_anomalies = np.sum(error_point > threshold_point)
        point_ratio = point_anomalies / len(error_point)

        # Contextual anomaly ratio
        context_anomalies = np.sum(rl_preds == 1)
        context_ratio = context_anomalies / len(rl_preds)

        # Final result logic
        types = []
        if point_ratio > 0.3:
            types.append("Point Anomaly")
        if context_ratio > 0.3:
            types.append("Contextual Anomaly")

        summary = "Anomalous" if types else "Normal"

        return jsonify({
            "summary": summary,
            "types": types
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Run Server ----------
if __name__ == "__main__":
    app.run(debug=True)
