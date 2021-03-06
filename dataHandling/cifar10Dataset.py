from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pickle

import matplotlib.pyplot as plt
from torchvision import transforms


class Cifar10Dataset(Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.data = list()
        self.transform = transform

        with open(path + "batches.meta", 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.labels = dict[b"label_names"]
            self.dataPerBatch = dict[b"num_cases_per_batch"]

        for data_batch_number in range(5):
            data_path = path + "data_batch_" + str(data_batch_number + 1)
            with open(data_path, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                self.data.append({"data" : dict[b"data"], "label": dict[b"labels"]})

    def __len__(self):
        return self.dataPerBatch * 5

    def __getitem__(self, item):
        data_set = int(item / self.dataPerBatch)
        index = item % self.dataPerBatch
        raw_img = self.data[data_set]["data"][index]
        img = Image.fromarray(self.dataset_img_to_numpy_img(raw_img))
        label = self.data[data_set]["label"][index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def dataset_img_to_numpy_img(self, data):
        img = np.array(data, dtype=np.uint8)
        img = img.reshape((3, 32, 32))
        return np.moveaxis(img, 0, -1)