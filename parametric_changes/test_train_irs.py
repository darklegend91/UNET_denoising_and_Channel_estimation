import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from unet_model import unet_model
import tensorflow as tf

# Prepare data function
def prepare_data(data):
    X_data = []
    Y_data = []
    for _, row in data.iterrows():
        try:
            users_array = np.array(eval(row['Users']))
            irs_array = np.array(eval(row['IRS']))
            combined_array = np.vstack([users_array, irs_array])
            X_data.append(combined_array)
            noise = np.random.normal(0, 0.1, combined_array.shape)
            Y_data.append(combined_array + noise)
        except Exception as e:
            print(f"Error processing row: {row}. Error: {e}")
            continue
    X_data = np.array(X_data).reshape(-1, *X_data[0].shape, 1)
    Y_data = np.array(Y_data).reshape(-1, *Y_data[0].shape, 1)
    return X_data, Y_data

# NMSE calculation function
def calculate_nmse(y_true, y_pred):
    return 10 * np.log10(np.mean(np.square(y_true - y_pred)) / np.mean(np.square(y_true)))

# Number of IRS elements to compare
irs_values = [8, 16, 32, 64]
results = []

for num_irs in irs_values:
    # Load dataset
    filename = f"generated_data_{num_irs}_irs_elements.xlsx"
    data = pd.read_excel(filename)

    # Prepare data
    X_data, Y_data = prepare_data(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

    # Define U-Net model
    input_shape = X_train.shape[1:]  # Exclude batch size
    model = unet_model(input_shape)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model and track NMSE at each epoch
    nmse_per_epoch = []
    for epoch in range(1, 21):  # Train for 20 epochs
        history = model.fit(X_train, Y_train, epochs=1, batch_size=16, validation_split=0.2, verbose=1)
        y_pred = model.predict(X_test)
        nmse = calculate_nmse(Y_test, y_pred)
        nmse_per_epoch.append(nmse)
        print(f"IRS Elements {num_irs}, Epoch {epoch}: NMSE {nmse:.4f} dB")

    # Store results
    results.append({
        'num_irs': num_irs,
        'epochs': list(range(1, 21)),
        'nmse_values': nmse_per_epoch
    })

    # Save model
    model.save(f"unet_model_{num_irs}_irs.h5")

# Save results to Excel
df_results = pd.DataFrame({
    'IRS Elements': [result['num_irs'] for result in results for _ in result['epochs']],
    'Epoch': [epoch for result in results for epoch in result['epochs']],
    'NMSE': [nmse for result in results for nmse in result['nmse_values']]
})
df_results.to_excel("nmse_vs_epoch_irs_results.xlsx", index=False)

# Plot results
plt.figure(figsize=(10, 6))
for result in results:
    plt.plot(result['epochs'], result['nmse_values'], marker='o', label=f"{result['num_irs']} IRS Elements")
plt.xlabel('Epochs')
plt.ylabel('NMSE (dB)')
plt.title('NMSE vs. Epoch for Different IRS Element Counts')
plt.legend()
plt.grid(True)
plt.savefig("nmse_vs_epoch_irs_plot.png")
plt.show()
