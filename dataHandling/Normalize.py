import torch
import numpy as np

class Normalize:

    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        data = (data / 256 * 2 )- 1
        return {'data': data,
                'label': label}