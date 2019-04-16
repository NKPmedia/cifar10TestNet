import pickle


class cifar10DataLoader():

    def loadData(self, path):
        with open(path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        