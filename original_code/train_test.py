import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from unet_model import unet_model
import tensorflow as tf

# Function to prepare data for U-Net
def prepare_data(data):
    X_data = []
    Y_data = []

    for _, row in data.iterrows():
        try:
            # Convert 'Users' and 'IRS' columns from string to numpy arrays
            users_array = np.array(eval(row['Users']))
            irs_array = np.array(eval(row['IRS']))

            # Combine Users and IRS arrays into one
            combined_array = np.vstack([users_array, irs_array])
            X_data.append(combined_array)

            # Add Gaussian noise to simulate denoising task
            noise = np.random.normal(0, 0.1, combined_array.shape)
            Y_data.append(combined_array + noise)

        except Exception as e:
            print(f"Error processing row: {row}. Error: {e}")
            continue

    # Convert to numpy arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    # Reshape data for U-Net model (add channel dimension)
    X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], X_data.shape[2], 1)
    Y_data = Y_data.reshape(Y_data.shape[0], Y_data.shape[1], Y_data.shape[2], 1)

    print(f"X_data shape: {X_data.shape}, Y_data shape: {Y_data.shape}")
    return X_data, Y_data

# NMSE calculation function
def calculate_nmse(y_true, y_pred):
    return 10 * np.log10(np.mean(np.square(y_true - y_pred)) / np.mean(np.square(y_true)))

# Load the dataset
data = pd.read_excel('generated_data.xlsx')

# Prepare data
X_data, Y_data = prepare_data(data)

# Ensure data is not empty
if X_data.size == 0 or Y_data.size == 0:
    raise ValueError("Prepared data is empty. Please check the data generation and preparation steps.")

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

# Create the U-Net model
input_shape = X_train.shape[1:]  # Exclude batch size
model = unet_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train the model
history = model.fit(X_train, Y_train, epochs=10, batch_size=16, validation_split=0.2, verbose=1)

# Save the trained model
model.save('channel_estimation_unet.h5')

# Evaluate the model at different SNR levels
snr_levels = [-10, -5, 0, 5, 10, 20]
nmse_values = []

for snr in snr_levels:
    noise_power = 10 ** (-snr / 10)  # Noise variance
    noise = np.random.normal(0, np.sqrt(noise_power), X_test.shape)
    noisy_data = X_test + noise  # Add noise to the test set
    y_pred = model.predict(noisy_data)
    nmse = calculate_nmse(Y_test, y_pred)
    nmse_values.append(nmse)
    print(f"SNR: {snr} dB, NMSE: {nmse:.4f} dB")

# # Save results to Excel
# results_df = pd.DataFrame({
#     'SNR': snr_levels,
#     'NMSE': nmse_values,
#     'Training Loss': history.history['loss'],
#     'Validation Loss': history.history['val_loss']
# })
# results_df.to_excel("training_results.xlsx", index=False)

# Save results for NMSE vs SNR
results_snr_df = pd.DataFrame({
    'SNR': snr_levels,
    'NMSE': nmse_values
})
results_snr_df.to_excel("nmse_vs_snr_results.xlsx", index=False)

# Save training and validation loss
results_loss_df = pd.DataFrame({
    'Epoch': range(1, len(history.history['loss']) + 1),
    'Training Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
})
results_loss_df.to_excel("training_loss_results.xlsx", index=False)


# Plot NMSE vs SNR
plt.figure(figsize=(10, 6))
plt.plot(snr_levels, nmse_values, marker='o', label='NMSE')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (dB)')
plt.title('NMSE vs SNR')
plt.grid(True)
plt.legend()
plt.savefig('nmse_vs_snr.png')
plt.close()

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss.png')
plt.close()

print("Training completed. Results saved to Excel and plots generated.")
