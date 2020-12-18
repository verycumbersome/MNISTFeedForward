import math
import numpy as np
import matplotlib.pyplot as plt
import gzip


def read_ubyte(filepath, is_img=True, image_size = 28, num_samples = 10000):
    f = gzip.open(filepath, "r")
    if (is_img):
        buf = f.read(image_size * image_size * num_samples)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_samples, image_size, image_size, 1)

    else:
        buf = f.read(num_samples)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    return(data.squeeze())


def ReLU(x):
    return(x if x > 0 else 0)


def normalize(array):
    return(array / np.sqrt(np.sum(array ** 2)))


class MnistDataLoader():
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return (len(self.images))

    def __getitem__(self, idx):
        return({
            "image":self.images[idx],
            "label":self.labels[idx],
        })


class Net():
    def __init__(self, train, test):
        self.train = train
        self.test = test

        self.l_size = 28
        self.n_layers = 3

        self.weights = np.random.rand(self.n_layers, self.l_size, self.l_size)
        self.biases = np.random.rand(self.l_size, 1)

    def forward(self, idx, layer):
        """z = Wx + b"""
        W = self.weights[layer]
        X = normalize(self.train[idx]["image"])
        b = self.biases

        z = np.matmul(X, W) + b
        z = z.reshape(784)

        print(z)
        print(z.shape)

        # Apply nonlinearity
        z = np.array([ReLU(x) for x in z]).reshape((28, 28))

        return

    def linear(self, input_size, output_size):
    #TODO Implement linear layer to go from size (784, 50) where
    # 784 is the input size and 50 is the output size

    def train(self):
        i = 0
        for j in range(self.n_layers):
            self.forward(i, j)
        # self.forward(2)
        # for item in self.train:
            # self.forward(2)


if __name__=="__main__":
    train_images = read_ubyte("data/train-images-idx3-ubyte.gz", is_img=True)
    test_images = read_ubyte("data/t10k-images-idx3-ubyte.gz", is_img=True)
    train_labels = read_ubyte("data/train-labels-idx1-ubyte.gz", is_img=False)
    test_labels = read_ubyte("data/t10k-labels-idx1-ubyte.gz", is_img=False)

    train_data = MnistDataLoader(train_images, train_labels)
    test_data = MnistDataLoader(test_images, test_labels)
    net = Net(train_data, test_data)
    net.forward(2, 0)
