import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Function to compute NMSE
def calculate_nmse(y_true, y_pred):
    return 10 * np.log10(np.mean((y_true - y_pred)**2) / np.mean(y_true**2))

# Function to train the model and compute NMSE at different SNRs
def train_and_evaluate(model, train_data, snr_levels=[-10, -5, 0, 5, 10, 20]):
    train_losses = []
    nmse_values = {snr: [] for snr in snr_levels}
    
    for epoch in range(10):  # Train for 10 epochs
        for X, Y in train_data:
            # Reshape data
            X = np.expand_dims(X, axis=-1)  # Add channel dimension
            Y = np.expand_dims(Y, axis=-1)  # Add channel dimension

            # Train the model
            model.fit(X, Y, epochs=1, batch_size=16)

            # Calculate NMSE for each SNR
            for snr in snr_levels:
                # Simulate noisy data for this SNR
                noise_std = np.sqrt(1 / (10**(snr / 10)))
                noisy_data = X + noise_std * np.random.randn(*X.shape)

                # Predict using the model
                y_pred = model.predict(noisy_data)

                # Calculate NMSE
                nmse = calculate_nmse(Y, y_pred)
                nmse_values[snr].append(nmse)

            # Store training loss
            train_losses.append(model.history.history['loss'][0])

    # Save results to an Excel file
    results_df = pd.DataFrame(nmse_values)
    results_df['Epoch'] = list(range(1, len(train_losses) + 1))
    results_df['Training Loss'] = train_losses
    results_df.to_excel("training_results.xlsx", index=False)

    return nmse_values, train_losses

# Train and evaluate the model
nmse_values, train_losses = train_and_evaluate(model, data)

# Plotting NMSE vs SNR
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

# Plotting Training Loss
plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Epoch')
plt.grid(True)
plt.savefig('training_loss.png')
plt.show()