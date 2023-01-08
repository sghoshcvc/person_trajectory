import torch
import scipy

from scipy.io import loadmat
import pandas as pd
import numpy as np
import os.path as osp
import os
from torch.utils.data import Dataset

# frames = pd.read_csv('Exp_1_run_1.tsv',delimiter='\t')
# contents = loadmat('Exp_1_run_1.mat')

    
# print('loaded')
# print(np.array(val_mat))
# print(np.array(val_mat).shape)
# print(trajectories[1].shape)

class Trajectory(Dataset):
    def __init__(self, data_folder):
        self.file_names = [
            osp.join(data_folder, x)
            for x in os.listdir(data_folder)
            if x.endswith('.tsv') 
        ]
        self.file_names = sorted(self.file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        filename = self.file_names[idx]
        f = open(filename,'r')
        lines = f.readlines()
        val_mat = []
        col_names = lines[10].split('\t')
        for line in lines[11:]:
            columns = line.split('\t')
            columns = [float(c) for c in columns]
            val_mat.append(columns)
        num_helmets = 9
        end = 0
        st = 0
        val_mat = np.array(val_mat)
        time = val_mat[:,:2]
        val_mat = val_mat[:,2:]
        trajectories = np.zeros((num_helmets,val_mat.shape[0],12))
        for x in range(num_helmets):
            end = st + 12
            trajectories[x] = val_mat[:,st:end]
            st = st +12
        return trajectories