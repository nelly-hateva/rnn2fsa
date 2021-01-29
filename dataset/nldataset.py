import os
import pickle

import numpy
import torch
from torch.utils import data as data_utils


class NLDataset(data_utils.Dataset):

    def __init__(self, dataset, device):
        data, length, labels = dataset
        self.data = torch.tensor(data).long().to(device)
        self.length = torch.tensor(length).to('cpu')
        self.labels = torch.tensor(labels).long().to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'x': self.data[idx],
            'length': self.length[idx],
            'y': self.labels[idx]
        }

    @staticmethod
    def load(folder):
        with open(os.path.join(folder, 'alphabet.dict'), 'rb') as f:
            alphabet = pickle.load(f)

        def load_split(folder_, split):
            return numpy.load(os.path.join(folder_, split + '.data.npy'), allow_pickle=True), \
                   numpy.load(os.path.join(folder_, split + '.length.npy'), allow_pickle=True), \
                   numpy.load(os.path.join(folder_, split + '.labels.npy'), allow_pickle=True)

        return load_split(folder, 'train'), load_split(folder, 'dev'), load_split(folder, 'test'), alphabet
