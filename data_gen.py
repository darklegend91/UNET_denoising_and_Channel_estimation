import numpy as np
import pandas as pd
import random

# Data generation function
def generate_data(num_samples, num_users, num_antennas, num_irs_elements):
    data = []
    for _ in range(num_samples):
        # Randomly generate channel data
        H_users = np.random.randn(num_users, num_antennas)  # User to antenna channel
        H_irs = np.random.randn(num_irs_elements, num_antennas)  # IRS to antenna channel
        
        # Combine the data into a single array
        X = np.concatenate((H_users, H_irs), axis=0)  # Combine the two channels

        # Generate noise with the same shape as X
        noise = np.random.randn(*X.shape)  # Noise with shape matching X

        Y = X + noise  # Denoised signal (noisy channel + noise)

        # Add data to the dataset
        data.append((X, Y))

    return data

# Function to save data to Excel
def save_data_to_excel(data, filename='generated_data.xlsx'):
    data_dict = {
        "Users": [d[0] for d in data],
        "IRS": [d[1] for d in data]
    }
    df = pd.DataFrame(data_dict)
    df.to_excel(filename, index=False)

# Generate 1000 samples with 4 users, 8 antennas, and 16 IRS elements
num_samples = 1000
num_users = 4
num_antennas = 8
num_irs_elements = 16

data = generate_data(num_samples, num_users, num_antennas, num_irs_elements)
save_data_to_excel(data)
