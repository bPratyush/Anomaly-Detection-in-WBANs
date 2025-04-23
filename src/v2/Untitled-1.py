# %%
import numpy as np
import pandas as pd
import gym
from gym import spaces
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO

# %%
df = pd.read_csv('/teamspace/studios/this_studio/examples/pamap2_HAR_raw.csv')

# %%
df=df.head(25000)

# %%
df.head()

# %%
df.describe()

# %%
sensor_columns = [col for col in df.columns if col not in ['timestamp', 'activityID', 'activity_name']]
sensor_data = df[sensor_columns]

# %%
#Linear Interpolation, Feature Engineering and Dropping NANs
sensor_data = sensor_data.interpolate(method='linear', axis=0)
sensor_data = sensor_data.dropna()

sensor_data['hand_acc_magnitude'] = np.sqrt(
    sensor_data['IMU_hand_3D_acceleration_1']**2 +
    sensor_data['IMU_hand_3D_acceleration_2']**2 +
    sensor_data['IMU_hand_3D_acceleration_3']**2
)
sensor_data['chest_acc_magnitude'] = np.sqrt(
    sensor_data['IMU_chest_3D_acceleration_1']**2 +
    sensor_data['IMU_chest_3D_acceleration_2']**2 +
    sensor_data['IMU_chest_3D_acceleration_3']**2
)
sensor_data['chest_gyro_magnitude'] = np.sqrt(
    sensor_data['IMU_chest_3D_gyroscope_1']**2 +
    sensor_data['IMU_chest_3D_gyroscope_2']**2 +
    sensor_data['IMU_chest_3D_gyroscope_3']**2
)
sensor_data['chest_mag_magnitude'] = np.sqrt(
    sensor_data['IMU_chest_3D_magnetometer_1']**2 +
    sensor_data['IMU_chest_3D_magnetometer_2']**2 +
    sensor_data['IMU_chest_3D_magnetometer_3']**2
)
sensor_data['ankle_acc_magnitude'] = np.sqrt(
    sensor_data['IMU_ankle_3D_acceleration_1']**2 +
    sensor_data['IMU_ankle_3D_acceleration_2']**2 +
    sensor_data['IMU_ankle_3D_acceleration_3']**2
)
sensor_data['ankle_gyro_magnitude'] = np.sqrt(
    sensor_data['IMU_ankle_3D_gyroscope_1']**2 +
    sensor_data['IMU_ankle_3D_gyroscope_2']**2 +
    sensor_data['IMU_ankle_3D_gyroscope_3']**2
)
sensor_data['ankle_mag_magnitude'] = np.sqrt(
    sensor_data['IMU_ankle_3D_magnetometer_1']**2 +
    sensor_data['IMU_ankle_3D_magnetometer_2']**2 +
    sensor_data['IMU_ankle_3D_magnetometer_3']**2
)
sensor_data['hand_acc_magnitude_4_6'] = np.sqrt(
    sensor_data['IMU_hand_3D_acceleration_4']**2 +
    sensor_data['IMU_hand_3D_acceleration_5']**2 +
    sensor_data['IMU_hand_3D_acceleration_6']**2
)
sensor_data['hand_gyro_magnitude'] = np.sqrt(
    sensor_data['IMU_hand_3D_gyroscope_1']**2 +
    sensor_data['IMU_hand_3D_gyroscope_2']**2 +
    sensor_data['IMU_hand_3D_gyroscope_3']**2
)
sensor_data['hand_mag_magnitude'] = np.sqrt(
    sensor_data['IMU_hand_3D_magnetometer_1']**2 +
    sensor_data['IMU_hand_3D_magnetometer_2']**2 +
    sensor_data['IMU_hand_3D_magnetometer_3']**2
)
sensor_data['chest_acc_magnitude_4_6'] = np.sqrt(
    sensor_data['IMU_chest_3D_acceleration_4']**2 +
    sensor_data['IMU_chest_3D_acceleration_5']**2 +
    sensor_data['IMU_chest_3D_acceleration_6']**2
)
sensor_data['ankle_acc_magnitude_4_6'] = np.sqrt(
    sensor_data['IMU_ankle_3D_acceleration_4']**2 +
    sensor_data['IMU_ankle_3D_acceleration_5']**2 +
    sensor_data['IMU_ankle_3D_acceleration_6']**2
)

# %%
cols_to_drop = [
    'IMU_chest_3D_acceleration_1','IMU_chest_3D_acceleration_2','IMU_chest_3D_acceleration_3',
    'IMU_chest_3D_acceleration_4','IMU_chest_3D_acceleration_5','IMU_chest_3D_acceleration_6',
    'IMU_hand_3D_acceleration_1','IMU_hand_3D_acceleration_2','IMU_hand_3D_acceleration_3',
    'IMU_hand_3D_acceleration_4','IMU_hand_3D_acceleration_5','IMU_hand_3D_acceleration_6',
    'IMU_chest_3D_gyroscope_1','IMU_chest_3D_gyroscope_2','IMU_chest_3D_gyroscope_3',
    'IMU_hand_3D_gyroscope_1','IMU_hand_3D_gyroscope_2','IMU_hand_3D_gyroscope_3',
    'IMU_chest_3D_magnetometer_1','IMU_chest_3D_magnetometer_2','IMU_chest_3D_magnetometer_3',
    'IMU_ankle_3D_acceleration_1','IMU_ankle_3D_acceleration_2','IMU_ankle_3D_acceleration_3',
    'IMU_ankle_3D_acceleration_4','IMU_ankle_3D_acceleration_5','IMU_ankle_3D_acceleration_6',
    'IMU_ankle_3D_gyroscope_1','IMU_ankle_3D_gyroscope_2','IMU_ankle_3D_gyroscope_3',
    'IMU_ankle_3D_magnetometer_1','IMU_ankle_3D_magnetometer_2','IMU_ankle_3D_magnetometer_3',
    'IMU_hand_3D_magnetometer_1','IMU_hand_3D_magnetometer_2','IMU_hand_3D_magnetometer_3'
]
sensor_data = sensor_data.drop(cols_to_drop, axis=1)

scaler = MinMaxScaler()
sensor_data_scaled = scaler.fit_transform(sensor_data)
sensor_data_scaled = pd.DataFrame(sensor_data_scaled, columns=sensor_data.columns)

sensor_data_scaled['timestamp'] = df['timestamp']
sensor_data_scaled['activityID'] = df['activityID']
sensor_data_scaled['activity_name'] = df['activity_name']

threshold_time = df['timestamp'].quantile(0.8)
support_set = sensor_data_scaled[sensor_data_scaled['timestamp'] < threshold_time]
query_set = sensor_data_scaled[sensor_data_scaled['timestamp'] >= threshold_time]

def drop_id_columns(df_):
    return df_.drop(['timestamp','activityID','activity_name'], axis=1)

support_features = drop_id_columns(support_set)
query_features = drop_id_columns(query_set)

# %%
sensor_data_scaled.head()

# %%
sensor_data_scaled.describe()

# %%
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data.iloc[i:i+seq_length].values)
    return np.array(sequences)

# %% [markdown]
# # LSTM-Autoencoder (Point Anomaly Detection)

# %%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam

# Define model parameters
seq_length_short = 50
encoding_dim = 32

# Prepare data
sensor_data_scaled_first_10000 = sensor_data_scaled.drop(['timestamp', 'activityID', 'activity_name'], axis=1)
X_seq_all = create_sequences(sensor_data_scaled_first_10000, seq_length_short)

# Extract dimensions
input_dim = X_seq_all.shape[2]
timesteps = X_seq_all.shape[1]

# Define autoencoder model
input_layer = Input(shape=(timesteps, input_dim))
encoded = LSTM(encoding_dim, activation='relu', return_sequences=False)(input_layer)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, activation='sigmoid', return_sequences=True)(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the model
history = autoencoder.fit(
    X_seq_all, X_seq_all,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Visualizing original vs reconstructed sequences
n_samples = 5  # Number of samples to visualize
sample_idx = np.random.choice(X_seq_all.shape[0], n_samples, replace=False)
X_sample = X_seq_all[sample_idx]
X_reconstructed = autoencoder.predict(X_sample)

# Plot some sequences
fig, axes = plt.subplots(n_samples, 1, figsize=(10, n_samples * 3))
for i in range(n_samples):
    axes[i].plot(X_sample[i].reshape(-1), label='Original', color='blue')
    axes[i].plot(X_reconstructed[i].reshape(-1), label='Reconstructed', linestyle='dashed', color='red')
    axes[i].set_title(f'Sample {i+1}')
    axes[i].legend()
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Obtain reconstruction error
X_pred_all = autoencoder.predict(X_seq_all)
reconstruction_error_all = np.mean(np.abs(X_pred_all - X_seq_all), axis=(1, 2))

# Set anomaly threshold at 95th percentile
threshold_all = np.percentile(reconstruction_error_all, 95)
anomalies_all = reconstruction_error_all > threshold_all

# Print total anomalies
print(f"Total anomalies (point detection): {np.sum(anomalies_all)}")

# Visualization: Reconstruction Error Distribution
plt.figure(figsize=(8, 5))
plt.hist(reconstruction_error_all, bins=50, alpha=0.75, color='blue')
plt.axvline(threshold_all, color='red', linestyle='dashed', label='Anomaly Threshold')
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.show()

# Visualization: Anomaly Detection Over Time
plt.figure(figsize=(12, 5))
plt.plot(reconstruction_error_all, label='Reconstruction Error', color='blue')
plt.axhline(threshold_all, color='red', linestyle='dashed', label='Threshold')
plt.scatter(np.where(anomalies_all)[0], reconstruction_error_all[anomalies_all], color='red', label='Anomalies', marker='x')
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Errors with Anomalies")
plt.legend()
plt.show()

# Visualization: Highlighting Anomalous Sequences
plt.figure(figsize=(10, 5))
plt.scatter(range(len(reconstruction_error_all)), reconstruction_error_all, c=anomalies_all, cmap='coolwarm', label="Anomaly Indicator")
plt.axhline(threshold_all, color='red', linestyle='dashed', label='Threshold')
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")
plt.title("Anomaly Visualization")
plt.colorbar(label="Anomalous (1) vs Normal (0)")
plt.legend()
plt.show()

# %% [markdown]
# # LSTM & Transformer Autoencoders for Sequential & Periodic Context

# %%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam

# Prepare LSTM autoencoder for sequential pass
X_seq_support_lstm = create_sequences(support_features, seq_length_short)
X_seq_query_lstm = create_sequences(query_features, seq_length_short)

# Extract dimensions
input_dim_lstm = X_seq_support_lstm.shape[2]
timesteps_lstm = X_seq_support_lstm.shape[1]

# Define LSTM autoencoder model
input_layer_lstm = Input(shape=(timesteps_lstm, input_dim_lstm))
encoded_lstm = LSTM(encoding_dim, activation='relu', return_sequences=False)(input_layer_lstm)
decoded_lstm = RepeatVector(timesteps_lstm)(encoded_lstm)
decoded_lstm = LSTM(input_dim_lstm, activation='sigmoid', return_sequences=True)(decoded_lstm)

autoencoder_lstm = Model(input_layer_lstm, decoded_lstm)
autoencoder_lstm.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the model
history_lstm = autoencoder_lstm.fit(
    X_seq_support_lstm, X_seq_support_lstm,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(history_lstm.history['loss'], label='Training Loss', color='blue')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('LSTM Autoencoder Loss Curve')
plt.legend()
plt.show()

# Obtain reconstruction error for support set
X_pred_support = autoencoder_lstm.predict(X_seq_support_lstm)
reconstruction_error_support = np.mean(np.abs(X_pred_support - X_seq_support_lstm), axis=(1, 2))

# Obtain reconstruction error for query set
X_pred_query = autoencoder_lstm.predict(X_seq_query_lstm)
reconstruction_error_query = np.mean(np.abs(X_pred_query - X_seq_query_lstm), axis=(1, 2))

# Histogram of reconstruction errors
plt.figure(figsize=(8, 5))
plt.hist(reconstruction_error_support, bins=50, alpha=0.7, label='Support Set', color='blue')
plt.hist(reconstruction_error_query, bins=50, alpha=0.7, label='Query Set', color='red')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Reconstruction Error Distribution')
plt.legend()
plt.show()

# Visualization: Original vs. Reconstructed Sequences
n_samples = 5
sample_idx = np.random.choice(X_seq_support_lstm.shape[0], n_samples, replace=False)
X_sample = X_seq_support_lstm[sample_idx]
X_reconstructed = autoencoder_lstm.predict(X_sample)

fig, axes = plt.subplots(n_samples, 1, figsize=(10, n_samples * 3))
for i in range(n_samples):
    axes[i].plot(X_sample[i].reshape(-1), label='Original', color='blue')
    axes[i].plot(X_reconstructed[i].reshape(-1), label='Reconstructed', linestyle='dashed', color='red')
    axes[i].set_title(f'Sample {i+1} - Support Set')
    axes[i].legend()
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Obtain predictions on query set
X_seq_query_pred_lstm = autoencoder_lstm.predict(X_seq_query_lstm)

# Compute reconstruction error
reconstruction_error_lstm = np.mean(np.abs(X_seq_query_pred_lstm - X_seq_query_lstm), axis=(1, 2))

# Set anomaly detection threshold (95th percentile)
threshold_lstm = np.percentile(reconstruction_error_lstm, 95)
anomalies_lstm = reconstruction_error_lstm > threshold_lstm

# Print total anomalies detected
print(f"Anomalies (LSTM, short window, query set): {np.sum(anomalies_lstm)}")

# Visualization 1: Histogram of Reconstruction Errors
plt.figure(figsize=(8, 5))
plt.hist(reconstruction_error_lstm, bins=50, alpha=0.75, color='blue')
plt.axvline(threshold_lstm, color='red', linestyle='dashed', label='Anomaly Threshold')
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution (Query Set)")
plt.legend()
plt.show()

# Visualization 2: Reconstruction Errors Over Time
plt.figure(figsize=(12, 5))
plt.plot(reconstruction_error_lstm, label='Reconstruction Error', color='blue')
plt.axhline(threshold_lstm, color='red', linestyle='dashed', label='Threshold')
plt.scatter(np.where(anomalies_lstm)[0], reconstruction_error_lstm[anomalies_lstm], color='red', label='Anomalies', marker='x')
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")
plt.title("Anomaly Detection Over Query Sequences")
plt.legend()
plt.show()

# Visualization 3: Scatter Plot Highlighting Anomalies
plt.figure(figsize=(10, 5))
plt.scatter(range(len(reconstruction_error_lstm)), reconstruction_error_lstm, c=anomalies_lstm, cmap='coolwarm', label="Anomaly Indicator")
plt.axhline(threshold_lstm, color='red', linestyle='dashed', label='Threshold')
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")
plt.title("Query Set Anomaly Visualization")
plt.colorbar(label="Anomalous (1) vs Normal (0)")
plt.legend()
plt.show()

# %%
# Periodic pass using a Transformer autoencoder
seq_length_long = 100
def create_sequences_long(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data.iloc[i:i+seq_length].values)
    return np.array(sequences)

X_seq_support_transformer = create_sequences_long(support_features, seq_length_long)
X_seq_query_transformer = create_sequences_long(query_features, seq_length_long)

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Define Transformer Autoencoder
def build_transformer_autoencoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Transformer Encoder
    encoder = layers.MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    encoder = layers.LayerNormalization()(encoder)
    encoder = layers.GlobalAveragePooling1D()(encoder)
    
    # LSTM Decoder
    decoder = layers.RepeatVector(input_shape[0])(encoder)
    decoder = layers.LSTM(128, return_sequences=True)(decoder)
    decoder = layers.Dense(input_shape[1], activation='linear')(decoder)
    
    model = tf.keras.Model(inputs, decoder)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Initialize and train the model
input_shape_transformer = (seq_length_long, X_seq_support_transformer.shape[2])
transformer_model = build_transformer_autoencoder(input_shape_transformer)

history_transformer = transformer_model.fit(
    X_seq_support_transformer, X_seq_support_transformer,
    epochs=20, batch_size=64, validation_split=0.2, verbose=1
)


# %%
# Plot Training Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(history_transformer.history['loss'], label='Training Loss', color='blue')
plt.plot(history_transformer.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Transformer-LSTM Autoencoder Loss Curve')
plt.legend()
plt.show()

# Obtain reconstruction error
X_seq_support_pred = transformer_model.predict(X_seq_support_transformer)
reconstruction_error_transformer = np.mean(np.abs(X_seq_support_pred - X_seq_support_transformer), axis=(1, 2))

# Histogram of Reconstruction Errors
plt.figure(figsize=(8, 5))
plt.hist(reconstruction_error_transformer, bins=50, alpha=0.75, color='blue')
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution (Support Set)")
plt.show()

# Visualizing Original vs. Reconstructed Sequences
n_samples = 5
sample_idx = np.random.choice(X_seq_support_transformer.shape[0], n_samples, replace=False)
X_sample = X_seq_support_transformer[sample_idx]
X_reconstructed = transformer_model.predict(X_sample)

fig, axes = plt.subplots(n_samples, 1, figsize=(10, n_samples * 3))
for i in range(n_samples):
    axes[i].plot(X_sample[i].reshape(-1), label='Original', color='blue')
    axes[i].plot(X_reconstructed[i].reshape(-1), label='Reconstructed', linestyle='dashed', color='red')
    axes[i].set_title(f'Sample {i+1} - Support Set')
    axes[i].legend()
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Obtain predictions on the query set
X_seq_query_pred_transformer = transformer_model.predict(X_seq_query_transformer)

# Compute reconstruction error
reconstruction_error_transformer = np.mean(np.abs(X_seq_query_pred_transformer - X_seq_query_transformer), axis=(1, 2))

# Set anomaly detection threshold (95th percentile)
threshold_transformer = np.percentile(reconstruction_error_transformer, 95)
anomalies_transformer = reconstruction_error_transformer > threshold_transformer

# Print total anomalies detected
print(f"Anomalies (Transformer, long window, query set): {np.sum(anomalies_transformer)}")

# %%
plt.figure(figsize=(8, 5))
plt.hist(reconstruction_error_transformer, bins=50, alpha=0.75, color='blue')
plt.axvline(threshold_transformer, color='red', linestyle='dashed', label='Anomaly Threshold')
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution (Query Set)")
plt.legend()
plt.show()

# Visualization 2: Reconstruction Errors Over Time
plt.figure(figsize=(12, 5))
plt.plot(reconstruction_error_transformer, label='Reconstruction Error', color='blue')
plt.axhline(threshold_transformer, color='red', linestyle='dashed', label='Threshold')
plt.scatter(np.where(anomalies_transformer)[0], reconstruction_error_transformer[anomalies_transformer], color='red', label='Anomalies', marker='x')
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")
plt.title("Anomaly Detection Over Query Sequences")
plt.legend()
plt.show()

# Visualization 3: Scatter Plot Highlighting Anomalies
plt.figure(figsize=(10, 5))
plt.scatter(range(len(reconstruction_error_transformer)), reconstruction_error_transformer, c=anomalies_transformer, cmap='coolwarm', label="Anomaly Indicator")
plt.axhline(threshold_transformer, color='red', linestyle='dashed', label='Threshold')
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")
plt.title("Query Set Anomaly Visualization")
plt.colorbar(label="Anomalous (1) vs Normal (0)")
plt.legend()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Ensure the same length for aggregation
min_len = min(len(reconstruction_error_lstm), len(reconstruction_error_transformer))
combined_reconstruction_error = reconstruction_error_lstm[:min_len] + reconstruction_error_transformer[:min_len]

# Normalize the aggregated anomaly scores
combined_reconstruction_error_normalized = (combined_reconstruction_error - np.min(combined_reconstruction_error)) / (np.max(combined_reconstruction_error) - np.min(combined_reconstruction_error))

# Set threshold for final anomalies (80% percentile)
combined_threshold = 0.8
final_anomalies = combined_reconstruction_error_normalized > combined_threshold

# Convert anomalies into binary labels for DRL training
final_labels = final_anomalies.astype(np.int32)

# Print total anomalies detected
print(f"Final anomalies after aggregation: {np.sum(final_anomalies)}")

# %%
# 1. Visualization: Histogram of Reconstruction Errors
plt.figure(figsize=(8, 5))
plt.hist(reconstruction_error_lstm, bins=50, alpha=0.6, label="LSTM Autoencoder", color='blue')
plt.hist(reconstruction_error_transformer, bins=50, alpha=0.6, label="Transformer Autoencoder", color='red')
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution (LSTM vs Transformer)")
plt.legend()
plt.show()

# 2. Visualization: Aggregated Anomaly Scores Over Time
plt.figure(figsize=(12, 5))
plt.plot(combined_reconstruction_error_normalized, label="Aggregated Anomaly Score", color='blue')
plt.axhline(combined_threshold, color='red', linestyle='dashed', label="Threshold (0.8)")
plt.scatter(np.where(final_anomalies)[0], combined_reconstruction_error_normalized[final_anomalies], color='red', label="Detected Anomalies", marker='x')
plt.xlabel("Sequence Index")
plt.ylabel("Normalized Anomaly Score")
plt.title("Aggregated Anomaly Scores Over Query Sequences")
plt.legend()
plt.show()

# 3. Visualization: Binary Labels for DRL Training
plt.figure(figsize=(12, 4))
plt.plot(final_labels, label="Final Anomaly Labels (0 = Normal, 1 = Anomaly)", linestyle='None', marker='o', markersize=3, color='black')
plt.xlabel("Sequence Index")
plt.ylabel("Anomaly Label")
plt.title("Final Binary Anomaly Labels for DRL Training")
plt.legend()
plt.show()


# %% [markdown]
# # DRL-based Anomaly Detection

# %%
class AnomalyDetectionEnv(gym.Env):
    def __init__(self, errors, labels):
        super(AnomalyDetectionEnv, self).__init__()
        self.errors = errors
        self.labels = labels  
        self.current_idx = 0
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self.current_idx = 0
        return np.array([self.errors[self.current_idx]], dtype=np.float32)

    def step(self, action):
        true_label = self.labels[self.current_idx]
        # Reward +1 for correct classification, -1 for misclassification
        reward = 1.0 if action == true_label else -1.0
        self.current_idx += 1
        done = self.current_idx >= len(self.errors)
        if not done:
            obs = np.array([self.errors[self.current_idx]], dtype=np.float32)
        else:
            obs = np.array([0.0], dtype=np.float32)
        return obs, reward, done, {}

# %%
# pip install seaborn

# %%
# pip install shimmy

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Define the environment (Assuming AnomalyDetectionEnv is already implemented)
env = AnomalyDetectionEnv(combined_reconstruction_error_normalized[:min_len], final_labels)

# DRL Training with DQN
model = DQN("MlpPolicy", env, verbose=1,
            learning_rate=5e-4,
            buffer_size=50000,
            exploration_fraction=0.1,
            learning_starts=1000,
            target_update_interval=100,
            batch_size=32)

# Train the model
model.learn(total_timesteps=15000)

# Test the trained model
obs = env.reset()
drl_actions = []
while True:
    action, _ = model.predict(obs, deterministic=True)
    drl_actions.append(action)
    obs, _, done, _ = env.step(action)
    if done:
        break

# Evaluate DQN performance
true_labels = final_labels[:len(drl_actions)]
accuracy = np.mean(np.array(drl_actions) == true_labels)
print(f"DQN agent accuracy on anomaly detection: {accuracy*100:.2f}%")

# %%
# Plot 1: DQN Anomaly Detection Decisions
plt.figure(figsize=(15, 6))
plt.plot(combined_reconstruction_error_normalized[:min_len], label='Combined Reconstruction Error', color='blue')
plt.axhline(y=combined_threshold, color='red', linestyle='--', label='Threshold')
plt.scatter(np.where(np.array(drl_actions) == 1)[0], 
            combined_reconstruction_error_normalized[:min_len][np.array(drl_actions) == 1],
            color='orange', marker='x', s=50, label='DQN: Anomaly')
plt.title('DQN Anomaly Detection Decisions')
plt.xlabel('Time Step')
plt.ylabel('Normalized Reconstruction Error')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Training Reward Curve (if available)
if "episode_rewards" in model.__dict__:
    plt.figure(figsize=(8, 5))
    plt.plot(model.__dict__["episode_rewards"], label="Episode Reward", color='blue')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Reward Curve")
    plt.legend()
    plt.grid()
    plt.show()

# Plot 3: Confusion Matrix
cm = confusion_matrix(true_labels, drl_actions)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - DQN Anomaly Detection")
plt.show()

# Plot 4: Anomaly Detection Decision Heatmap
plt.figure(figsize=(15, 2))
sns.heatmap([drl_actions], cmap="coolwarm", cbar=False, xticklabels=False, yticklabels=["DQN Actions"])
plt.xlabel("Time Step")
plt.title("DQN Anomaly Detection Decision Heatmap")
plt.show()

# Plot 5: ROC Curve and AUC Score
fpr, tpr, _ = roc_curve(true_labels, drl_actions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - DQN Anomaly Detection")
plt.legend()
plt.grid()
plt.show()

# %%
num_anomalies_detected = np.sum(np.array(drl_actions) == 1)
print(f"Total anomalies detected by the DQN agent: {num_anomalies_detected}")


