import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from Net.Network import Network
from Net.Trainer import Trainer
from dataHandling.Normalize import Normalize
from dataHandling.ToTensor import ToTensor
from dataHandling.cifar10Dataset import Cifar10Dataset
import matplotlib.pyplot as plt
import numpy as np

from dataHandling.cifar10TestDataset import Cifar10TestDataset

transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataSet = Cifar10Dataset("/work/scratch/kruse/Projekts/TorchTest/data/cifar-10-batches-py/", transform=transform_train)
test_dataSet = Cifar10TestDataset("/work/scratch/kruse/Projekts/TorchTest/data/cifar-10-batches-py/", transform=transform_train)


sample = dataSet[1]

dataloader = DataLoader(dataSet, batch_size=8, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataSet, batch_size=8, shuffle=True, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Network()
trainer = Trainer(net, device)
trainer.train(dataloader)

correct = 0
total = 0
with torch.no_grad():
    for i, sample in enumerate(test_dataloader):
        data, label = sample[0], sample[1]
        data, label = data.to(device), label.to(device)
        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))