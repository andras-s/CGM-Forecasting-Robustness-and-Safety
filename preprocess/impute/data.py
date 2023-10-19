import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.utils import shuffle


class GaussianProcessDataset(Dataset):
    """
    Dataset for the training of the Gaussian process
    """
    def __init__(self, data, col_inputs='relative_time', col_target='glucose_value'):

        self.inputs = [torch.tensor(df[col_inputs].values, dtype=torch.float32) for df in data]
        self.target = [torch.tensor(df[col_target].values, dtype=torch.float32) for df in data]

        assert np.all(np.array([t.shape[0] for t in self.inputs]) == \
                      np.array([t.shape[0] for t in self.target])), "Inputs and targets have different lengths"

        # padding needs to be done due to variable input length
        self.lengths = torch.tensor([len(t) for t in self.inputs], dtype=torch.int)
        max_length = self.lengths.max().item()

        self.inputs = torch.stack([F.pad(t, (0, max_length - len(t))) for t in self.inputs])
        self.target = torch.stack([F.pad(t, (0, max_length - len(t))) for t in self.target])

    def shuffle(self):
        self.inputs, self.target = shuffle(self.inputs, self.target)

    def random_sample(self):
        ix = np.random.choice(len(self.inputs))
        l = self.lengths[ix]
        return self.inputs[ix][:l], self.target[ix][:l]

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, i):
        return self.inputs[i], self.target[i], self.lengths[i]
