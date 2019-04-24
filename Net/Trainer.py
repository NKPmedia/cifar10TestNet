from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD

from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

writer = SummaryWriter(comment='Hoz_Sched06')

class Trainer():

    def __init__(self, net, device):
        self.epoch = 8
        self.lr = 0.0001
        self.lambda_factor = 0.6
        self.net = net
        self.device = device

        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(net.parameters(), lr=self.lr)

        lambda1 = lambda epoch: self.lambda_factor ** epoch
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=[lambda1])

    def train(self, trainloader):
        self.net.to(self.device)
        running_loss = 0
        logging_i = 0
        for epoch_i in range(self.epoch):

            self.scheduler.step()

            for i, sample in enumerate(trainloader):
                data, label = sample[0], sample[1]
                data, label = data.to(self.device), label.to(self.device)

                self.optimizer.zero_grad()
                out = self.net(data)
                loss = self.criterion(out, label)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 200 == 199:
                    writer.add_scalar('data/loss', running_loss / 200, logging_i)
                    logging_i += 1
                    running_loss = 0.0
        print('Finished Training')