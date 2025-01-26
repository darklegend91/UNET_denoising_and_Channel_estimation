# import numpy as np
# import pandas as pd

# # Data generation function
# def generate_data(num_samples, num_users, num_antennas, num_irs_elements):
#     data = []
#     for _ in range(num_samples):
#         H_users = np.random.randn(num_users, num_antennas)  # User to antenna channel
#         H_irs = np.random.randn(num_irs_elements, num_antennas)  # IRS to antenna channel
#         X = np.concatenate((H_users, H_irs), axis=0)  # Combine channels
#         noise = np.random.randn(*X.shape)  # Generate noise
#         Y = X + noise  # Noisy data
#         data.append((X, Y))
#     return data

# # Save data to Excel
# def save_data_to_excel(data, filename):
#     users_data = []
#     irs_data = []
#     for X, Y in data:
#         users_data.append(X[:4, :].tolist())  # Users data
#         irs_data.append(X[4:, :].tolist())   # IRS data

#     df = pd.DataFrame({'Users': users_data, 'IRS': irs_data})
#     df.to_excel(filename, index=False)

# # Generate datasets for different user values
# num_samples = 1000
# num_antennas = 8
# num_irs_elements = 16
# user_values = [2, 4, 6]

# for num_users in user_values:
#     data = generate_data(num_samples, num_users, num_antennas, num_irs_elements)
#     filename = f"generated_data_{num_users}_users.xlsx"
#     save_data_to_excel(data, filename)
#     print(f"Data for {num_users} users saved to {filename}.")

# import numpy as np
# import pandas as pd

# def generate_data(num_samples, num_users, num_irs_elements, num_antennas):
#     """
#     Generate synthetic dataset for channel estimation with varying number of antennas.

#     Args:
#         num_samples (int): Number of samples to generate.
#         num_users (int): Number of users.
#         num_irs_elements (int): Number of IRS elements.
#         num_antennas (int): Number of antennas.

#     Returns:
#         pd.DataFrame: Dataframe containing generated user and IRS data.
#     """
#     data = []

#     for _ in range(num_samples):
#         # Simulate random user data (channel coefficients)
#         users = np.random.randn(num_users, num_antennas).tolist()

#         # Simulate IRS reflection coefficients
#         irs = np.random.randn(num_irs_elements, num_antennas).tolist()

#         data.append({
#             'Users': str(users),   # Convert to string for easier saving in Excel
#             'IRS': str(irs),       # Convert to string
#             'Antennas': num_antennas
#         })

#     df = pd.DataFrame(data)
#     return df

# # Parameters for data generation
# num_samples = 1000  # Number of data samples
# num_users = 4       # Fixed number of users
# num_irs_elements = 16  # Fixed number of IRS elements
# antenna_values = [2, 4, 8, 16]  # Varying number of antennas

# # Generate and save datasets for different antenna values
# for num_antennas in antenna_values:
#     df = generate_data(num_samples, num_users, num_irs_elements, num_antennas)
#     filename = f'generated_data_{num_antennas}_antennas.xlsx'
#     df.to_excel(filename, index=False)
#     print(f"Generated and saved data for {num_antennas} antennas to {filename}")

import numpy as np
import pandas as pd

def generate_data(num_samples, num_users, num_irs_elements, num_antennas):
    """
    Generate synthetic dataset for channel estimation with varying number of IRS elements.

    Args:
        num_samples (int): Number of samples to generate.
        num_users (int): Number of users.
        num_irs_elements (int): Number of IRS elements.
        num_antennas (int): Number of antennas.

    Returns:
        pd.DataFrame: Dataframe containing generated user and IRS data.
    """
    data = []

    for _ in range(num_samples):
        # Simulate random user data (channel coefficients)
        users = np.random.randn(num_users, num_antennas).tolist()

        # Simulate IRS reflection coefficients
        irs = np.random.randn(num_irs_elements, num_antennas).tolist()

        data.append({
            'Users': str(users),   # Convert to string for easier saving in Excel
            'IRS': str(irs),       # Convert to string
            'IRS_Elements': num_irs_elements
        })

    df = pd.DataFrame(data)
    return df

# Parameters for data generation
num_samples = 1000  # Number of data samples
num_users = 4       # Fixed number of users
num_antennas = 8    # Fixed number of antennas
irs_values = [8, 16, 32, 64]  # Varying number of IRS elements

# Generate and save datasets for different IRS element values
for num_irs_elements in irs_values:
    df = generate_data(num_samples, num_users, num_irs_elements, num_antennas)
    filename = f'generated_data_{num_irs_elements}_irs_elements.xlsx'
    df.to_excel(filename, index=False)
    print(f"Generated and saved data for {num_irs_elements} IRS elements to {filename}")
