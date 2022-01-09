import torch
import numpy as np
import os
import gzip
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def set_seeds(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return

class MNIST(Dataset):

    ### Overwriting some functions of Dataset build in class
    def __init__(self, dataset_path, device):

        data = torch.load(dataset_path)
        self.imgs = data[0].data.cpu().numpy().reshape(len(data[0]), -1) / 255
        # self.labels = data[1]
        self.device = device

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        current_img = self.imgs[idx]

        ## to binary value observation (according to Yuri et al)
        current_binarized_img = np.random.binomial(1, current_img).astype('float32')

        # return current_binarized_img.to(self.device)#, current_label.to(self.device)
        return current_binarized_img


class MNIST_wlabels(Dataset):

    ### Overwriting some functions of Dataset build in class
    def __init__(self, dataset_path, device):

        data = torch.load(dataset_path)
        self.imgs = data[0].data.cpu().numpy().reshape(len(data[0]), -1) / 255
        self.labels = data[1]
        self.device = device

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        current_img = self.imgs[idx]
        current_label = self.labels[idx]

        ## to binary value observation (according to Yuri et al)
        current_binarized_img = np.random.binomial(1, current_img).astype('float32')

        # return current_binarized_img.to(self.device)#, current_label.to(self.device)
        return current_binarized_img, current_label