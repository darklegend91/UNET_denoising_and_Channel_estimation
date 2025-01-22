import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from unet_model import unet_model  # Import the model definition

# Load generated dataset
data = pd.read_excel('generated_data.xlsx')

# Prepare data for training
def prepare_data(data):
    X_data = []
    Y_data = []

    for row in data.itertuples():
        X_data.append(np.concatenate([row.Users, row.IRS], axis=0))
        Y_data.append(np.concatenate([row.Users + np.random.randn(*row.Users.shape), row.IRS], axis=0))

    return np.array(X_data), np.array(Y_data)

X_train, Y_train = prepare_data(data)

# Define input shape based on the data dimensions
input_shape = (X_train.shape[1], 8, 1)

# Create and compile the model
model = unet_model(input_shape)

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=16)

# Save the model
model.save('channel_estimation_unet.h5')

# Load the trained model
model = load_model('channel_estimation_unet.h5')

# Function to compute NMSE
def calculate_nmse(y_true, y_pred):
    return 10 * np.log10(np.mean((y_true - y_pred)**2) / np.mean(y_true**2))

# Evaluate the model at different SNR levels
snr_levels = [-10, -5, 0, 5, 10, 20]
nmse_values = {snr: [] for snr in snr_levels}
train_losses = []

for epoch in range(10):
    for X_batch, Y_batch in zip(X_train, Y_train):
        # Simulate noisy data for each SNR level and calculate NMSE
        for snr in snr_levels:
            noise_std = np.sqrt(1 / (10**(snr / 10)))
            noisy_data = X_batch + noise_std * np.random.randn(*X_batch.shape)
            y_pred = model.predict(noisy_data)
            nmse = calculate_nmse(Y_batch, y_pred)
            nmse_values[snr].append(nmse)

    # Save training loss
    train_losses.append(model.history.history['loss'][-1])

# Save results to Excel
results_df = pd.DataFrame(nmse_values)
results_df['Epoch'] = list(range(1, len(train_losses) + 1))
results_df['Training Loss'] = train_losses
results_df.to_excel("training_results.xlsx", index=False)

# Plot NMSE vs SNR
plt.figure()
for snr, values in nmse_values.items():
    plt.plot(range(1, len(values) + 1), values, label=f'SNR = {snr} dB')
plt.xlabel('Epoch')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.title('NMSE vs Epoch')
plt.grid(True)
plt.savefig('nmse_vs_epoch.png')
plt.show()

# Plot Training Loss vs Epoch
plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Epoch')
plt.grid(True)
plt.savefig('training_loss.png')
plt.show()
