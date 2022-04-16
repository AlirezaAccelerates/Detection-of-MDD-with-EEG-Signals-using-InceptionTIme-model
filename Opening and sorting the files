#pip install mne                                     #if you don't have mne
import mne
import numpy as np
from os.path import exists
from sklearn import preprocessing

H_num =                                              #Number of healthy cases
MDD_num =                                            ##Number of MDD cases
v = []
Data = []
Label = []
HorMDD = ['H', 'MDD']
eye = ['O','C']

for x in HorMDD:
    for y in eye:
        if  x == 'H':
            for i in range(1,H_num):
                if exists('/your directory/H S{} E{}.edf'.format(i, y)):
                    file = '/your directory/H S{} E{}.edf'.format(i, y)

                    data = mne.io.read_raw_edf(file)
                    raw_data = data.get_data()
                    num_rows, num_cols = raw_data.shape
                    if num_rows > 19:
                        print(num_rows)
                        raw_data = np.delete(raw_data, 19 , 0)                        
                    if len(raw_data[1]) < 61440:
                        v.append(file)
                        continue
                    raw_data = raw_data[:,:61440]
                    Data.append(raw_data)
                    Label.append(0)

                    np.save('/your directory/Data/H S{} E{}.npy'.format(i, y), raw_data)
                    np.save('/your directory/Label/H S{} E{}.npy'.format(i, y), 0)               
                else:
                    continue
            
            
        if  x == 'MDD':
            for j in range(1,MDD_num):
                if exists('/your directory/MDD S{} E{}.edf'.format(j, y)):
                    file = '/your directory/MDD S{} E{}.edf'.format(j, y)
            
                    data = mne.io.read_raw_edf(file)
                    raw_data = data.get_data()
                    num_rows, num_cols = raw_data.shape
                    
                    if num_rows > 19:
                       print(num_rows)
                       raw_data = np.delete(raw_data, 19 , 0)
                    if len(raw_data[1]) < 61440:
                        v.append(file)
                        continue
                    raw_data = raw_data[:,:61440]
                    Data.append(raw_data)
                    Label.append(1)

                    np.save('/your directory/Data/MDD S{} E{}.npy'.format(j, y), raw_data)
                    np.save('/your directory/MDD S{} E{}.npy'.format(j, y), 1)
                else:
                    continue
                    
Data = np.asarray(Data)
Label = np.asarray(Label)
