import numpy as np
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader, Dataset

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels, weights):
        self.data = data
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.weights[idx]


labels_dict = {
    '0': 'mWJJ',
    '1': 'mJJ',
    '2': 'mWJ1',
    '3': 'mWJ2',
    '4': 'mJ1',
    '5': 'mJ2',
    '6': 'mWJh',
    '7': 'mWJl',
    '8': 'mJh',
    '9': 'mJl',
    '10': 'score11', '11': 'score12', '12': 'score13', 
    '13': 'score21', '14': 'score22', '15': 'score23', 
    '16': 'weight'
}

files = {
    'Wjj-3400': './WDijetsC/Wjj-3400.dat',
    'WWj-3400':'./WDijetsC/WWj-3400.dat',
    'Zp3500-H1500_H300_Wl3Wh-3400': './WDijetsC/Zp3500-H1500_H300_Wl3Wh-3400.dat',
    'Zp3500-Hc2000_HWlWh1-3400': './WDijetsC/Zp3500-Hc2000_HWlWh1-3400.dat',
    'Zp3500-Hc2000_HWlWh2-3400': './WDijetsC/Zp3500-Hc2000_HWlWh2-3400.dat',
    'Zp3500-Hc800_HWlWh1-3400': './WDijetsC/Zp3500-Hc800_HWlWh1-3400.dat',
    'Zp3500-Hc800_HWlWh2-3400': './WDijetsC/Zp3500-Hc800_HWlWh2-3400.dat',
}
N_process = {
    "Wjj-3400" : 131217,
    "WWj-3400" : 16778,
    "Zp3500-H1500_H300_Wl3Wh-3400" : 19791,
    "Zp3500-Hc2000_HWlWh1-3400" : 14733,
    "Zp3500-Hc2000_HWlWh2-3400" : 15108,
    "Zp3500-Hc800_HWlWh1-3400" : 16009,
    "Zp3500-Hc800_HWlWh2-3400" : 11226,
}
'''
mean_bkg = np.array([3.45018147e+03, 2.69449879e+03, 1.72603587e+03, 8.31362396e+02,
       1.39907697e+02, 1.28568662e+02, 1.26392708e+03, 1.29347119e+03,
       1.95814074e+02, 7.26622857e+01, 2.79641656e-01, 3.65593244e-01,
       3.54765100e-01, 2.34072543e-01, 3.80639165e-01, 3.85288293e-01])
std_bkg = np.array([2.87811809e+01, 5.80024576e+02, 7.13128997e+02, 5.12112434e+02,
       1.25583777e+02, 1.08349663e+02, 8.06288917e+02, 7.21452825e+02,
       1.25130941e+02, 6.58353058e+01, 2.42351089e-01, 2.23554281e-01,
       2.19622179e-01, 2.24948674e-01, 2.34665220e-01, 2.25984741e-01,])
'''
mean_bkg = np.array([3.44906220e+03, 2.52818294e+03, 1.70194996e+03, 1.06099284e+03,
       9.58717480e+01, 8.25535557e+01, 1.32370009e+03, 1.43924271e+03,
       1.27392472e+02, 5.10328316e+01, 2.45745297e-01, 3.85769696e-01,
       3.68485006e-01, 2.24818117e-01, 3.92671569e-01, 3.82510314e-01])

std_bkg = np.array([2.88989014e+01, 6.41144830e+02, 7.76989975e+02, 7.81649385e+02,
       9.05552104e+01, 7.33170952e+01, 8.65668016e+02, 8.14890197e+02,
       9.25298454e+01, 4.67670833e+01, 2.18737548e-01, 2.36636183e-01,
       2.12549487e-01, 2.09482159e-01, 2.41167239e-01, 2.13918977e-01])

def standardize(dataset, mean_all, std_all):
    dataset_new = np.copy(dataset)
    for j in range(dataset.shape[1]):
        mean, std = mean_all[j], std_all[j]
        vec  = dataset[:, j]
        if np.min(vec) < 0:
            vec = vec- mean
            vec = vec *1./ std
        elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.                                                
            vec = vec *1./ mean
        dataset_new[:, j] = vec
    return dataset_new

def inv_standardize(dataset, mean_all, std_all):
    dataset_new = np.copy(dataset)
    for j in range(dataset.shape[1]):
        mean, std = mean_all[j], std_all[j]
        vec  = dataset[:, j]
        if np.min(vec) < 0:
            dataset_new[:, j] = dataset[:, j] * std + mean
        elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1                                                       
            dataset_new[:, j] = dataset[:, j] * mean
    return dataset_new



def candidate_sigma(data, perc=90):
    # this function estimates the width of the gaussian kernel.                                                                   \
                                                                                                                               
    # use on a (small) sample of reference data (standardize first if necessary)                                                  \
                                                                                                                                
    pairw = pdist(data)
    return np.around(np.percentile(pairw,perc),2)
