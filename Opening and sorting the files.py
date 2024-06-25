import numpy as np
import mne
from os.path import exists
from sklearn import preprocessing

# Define the number of healthy and MDD (Major Depressive Disorder) cases
H_num =  # Number of healthy cases
MDD_num =  # Number of MDD cases

# Initialize lists to store data and labels
v = []
Data = []
Label = []

# Define the categories for healthy and MDD cases
HorMDD = ['H', 'MDD']
# Define the eye states: Open (O) and Closed (C)
eye = ['O', 'C']

# Iterate over each category (healthy and MDD)
for x in HorMDD:
    # Iterate over each eye state
    for y in eye:
        # Process healthy cases
        if x == 'H':
            for i in range(1, H_num):
                # Check if the file exists
                if exists('/your directory/H S{} E{}.edf'.format(i, y)):
                    file = '/your directory/H S{} E{}.edf'.format(i, y)

                    # Read the EEG data from the .edf file
                    data = mne.io.read_raw_edf(file)
                    raw_data = data.get_data()
                    num_rows, num_cols = raw_data.shape

                    # Ensure the number of rows is correct
                    if num_rows > 19:
                        print(num_rows)
                        raw_data = np.delete(raw_data, 19, 0)

                    # Ensure the data length is sufficient
                    if len(raw_data[1]) < 61440:
                        v.append(file)
                        continue

                    # Trim the data to the desired length
                    raw_data = raw_data[:, :61440]
                    Data.append(raw_data)
                    Label.append(0)  # Label 0 for healthy cases

                    # Save the processed data and labels
                    np.save('/your directory/Data/H S{} E{}.npy'.format(i, y), raw_data)
                    np.save('/your directory/Label/H S{} E{}.npy'.format(i, y), 0)
                else:
                    continue

        # Process MDD cases
        if x == 'MDD':
            for j in range(1, MDD_num):
                # Check if the file exists
                if exists('/your directory/MDD S{} E{}.edf'.format(j, y)):
                    file = '/your directory/MDD S{} E{}.edf'.format(j, y)

                    # Read the EEG data from the .edf file
                    data = mne.io.read_raw_edf(file)
                    raw_data = data.get_data()
                    num_rows, num_cols = raw_data.shape

                    # Ensure the number of rows is correct
                    if num_rows > 19:
                        print(num_rows)
                        raw_data = np.delete(raw_data, 19, 0)

                    # Ensure the data length is sufficient
                    if len(raw_data[1]) < 61440:
                        v.append(file)
                        continue

                    # Trim the data to the desired length
                    raw_data = raw_data[:, :61440]
                    Data.append(raw_data)
                    Label.append(1)  # Label 1 for MDD cases

                    # Save the processed data and labels
                    np.save('/your directory/Data/MDD S{} E{}.npy'.format(j, y), raw_data)
                    np.save('/your directory/Label/MDD S{} E{}.npy'.format(j, y), 1)
                else:
                    continue

# Convert the lists to numpy arrays
Data = np.asarray(Data)
Label = np.asarray(Label)
