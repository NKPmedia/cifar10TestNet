from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = Conv2d(3, 32, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = Conv2d(32, 32, 3, padding=1, stride=2)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = Conv2d(32, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = Conv2d(64, 64, 3, padding=1, stride=2)
        self.conv4_bn = nn.BatchNorm2d(64)

        self.conv5 = Conv2d(64, 128, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = Conv2d(128, 128, 3, padding=1, stride=2)
        self.conv6_bn = nn.BatchNorm2d(128)

        self.conv7 = Conv2d(128, 256, 3, padding=1)
        self.conv7_bn = nn.BatchNorm2d(256)
        self.conv8 = Conv2d(256, 256, 3, padding=1, stride=2)
        self.conv8_bn = nn.BatchNorm2d(256)

        self.fc1 = Linear(2*2*256, 10)


    def forward(self, x):
        x = self.conv1_bn(F.leaky_relu(self.conv1(x)))
        x = self.conv2_bn(F.leaky_relu(self.conv2(x)))

        x = self.conv3_bn(F.leaky_relu(self.conv3(x)))
        x = self.conv4_bn(F.leaky_relu(self.conv4(x)))

        x = self.conv5_bn(F.leaky_relu(self.conv5(x)))
        x = self.conv6_bn(F.leaky_relu(self.conv6(x)))

        x = self.conv7_bn(F.leaky_relu(self.conv7(x)))
        x = self.conv8_bn(F.leaky_relu(self.conv8(x)))

        x = x.view(-1, 256 * 2 * 2)
        x = F.softmax(self.fc1(x))
        return x