import torch
import numpy as np

class ToTensor:

    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        data = np.moveaxis(data, 0, -1)
        data = np.moveaxis(data, 0, -1)
        return {'data': torch.from_numpy(data),
                'label': label}