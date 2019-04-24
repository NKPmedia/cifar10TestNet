from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pickle

class Cifar10TestDataset(Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.data = list()
        self.transform = transform

        with open(path + "batches.meta", 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.labels = dict[b"label_names"]
            self.dataPerBatch = dict[b"num_cases_per_batch"]

        data_path = path + "test_batch"
        with open(data_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.data.append({"data" : dict[b"data"], "label": dict[b"labels"]})

    def __len__(self):
        return self.dataPerBatch

    def __getitem__(self, item):
        raw_img = self.data[0]["data"][item]
        img = Image.fromarray(self.dataset_img_to_numpy_img(raw_img))
        label = self.data[0]["label"][item]
        if self.transform:
            img = self.transform(img)
        return img, label

    def dataset_img_to_numpy_img(self, data):
        img = np.array(data, dtype=np.uint8)
        img = img.reshape((3, 32, 32))
        return np.moveaxis(img, 0, -1)