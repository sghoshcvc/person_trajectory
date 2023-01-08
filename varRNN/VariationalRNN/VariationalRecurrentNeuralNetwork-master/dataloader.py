import torch
import scipy

from scipy.io import loadmat
import pandas as pd
import numpy as np
import os.path as osp
import os
from torch.utils.data import Dataset
import scipy.interpolate as interpolate

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
        self.times = 1024 *25
        self.segments = self.create_segments()
        # print('loaded')
        # self.cal_min_timestamps()

    def create_segments(self):
        segments = []
        for idx in range(len(self.file_names)):
            filename = self.file_names[idx]
            f = open(filename,'r')
            lines = f.readlines()
            val_mat = []
            for line in lines[11:]:
                columns = line.split('\t')
                columns = [float(c) for c in columns]
                val_mat.append(columns)
            val_mat = np.array(val_mat)
            time = val_mat[:,1]
            val_mat = val_mat[:,2:]
            step = time[-1] / float(self.times)
            self.timestamps = np.arange(0,time[-1],step)
            st = 0
            num_helmets = 9
            # trajectories = np.zeros((num_helmets,len(self.timestamps),12))
            trajectories = []
            for x in range(num_helmets):
                end = st+12
                # trajectories[x] = val_mat[:,st:end]
                trajs_fn = interpolate.interp1d(time,val_mat[:,st:end], axis = 0)
                st = st+12
                trajs = trajs_fn(self.timestamps)
                trajectories.append(trajs.reshape(25,1024,12))
            trajectories = np.array(trajectories)
            
            segments.append(trajectories)
        segments = np.array(segments)
        # flatten as num_samples x 1024 x 12
        segments = segments.reshape(-1,1024,12)

        return segments

    def cal_min_timestamps(self):
        num_timestamps = []
        for idx in range(len(self.file_names)):
            filename = self.file_names[idx]
            f = open(filename,'r')
            lines = f.readlines()
            num_timestamps.append(len(lines)-11)
        min_timestapmps = np.min(np.array(num_timestamps))
        return min_timestapmps

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        trajectory = self.segments[idx]
        
        return trajectory