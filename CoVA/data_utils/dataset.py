import os
import torch
from torch.utils.data import Dataset

class Dataset_CEU(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith('.tokens.pt')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.data_dir, self.filenames[idx])).cpu(), self.filenames[idx].split(".")[0]
    
class Dataset_Dx(Dataset):
    def __init__(self, data_dir, out_filename=False):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.cv.pt')]
        self.out_filename = out_filename

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        splited = filename.split(".")
        if "UNKNOWN" in splited[0]:
            label = None
        else:
            label = "ALS" in splited[0]
            label = torch.tensor(int(label))
        data, label = torch.load(filename).cpu(), label

        if self.out_filename:
            return data, label, filename
        return data, label