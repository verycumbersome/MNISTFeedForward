import math
import random
import numpy as np
import matplotlib.pyplot as plt
import gzip
from dataclasses import dataclass


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


def sigmoid(x):
    return(1 / (1 + (math.e ** (-x))))


def normalize(array):
    return(array / np.sqrt(np.sum(array ** 2)))


# def softmax(array):
    # return(array / np.sqrt(np.sum(array ** 2)))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


class MnistDataLoader():
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return (len(self.images))

    def __getitem__(self, idx):
        return({
            "image":self.images[idx],
            "label":int(self.labels[idx]),
        })


@dataclass
class LinearLayer():
    in_size: int
    out_size: int

    def __post_init__(self):
        # For each node in output layer, generate empty weights and biases
        self.weights = np.random.randn(self.out_size, self.in_size) * \
                np.sqrt(2 / self.in_size)
        self.biases = np.random.uniform(0, 1, self.out_size)


    def calc(self, x):
        """Function: z = Wx + b"""
        layer_output = []

        for i in range(self.out_size):
            # z = Wx + b
            z = np.dot(self.weights[i], x) + self.biases[i]
            layer_output.append(z)

        return(np.array(layer_output))

    def backprop():
        pass


class Net():
    def __init__(self, train, test):
        self.L1 = LinearLayer(784, 50)
        self.L2 = LinearLayer(50, 10)

    def forward(self, x):
        """Get prediction from nueral net"""
        x = x.reshape(784)
        x = self.L1.calc(x)
        x = np.array([ReLU(n) for n in x])

        x = self.L2.calc(x)
        x = np.array([ReLU(n) for n in x])

        x = np.argmax(x)

        return x

def loss(results):
    for i in range(10):
        pass


if __name__=="__main__":
    train_images = read_ubyte("data/train-images-idx3-ubyte.gz", is_img=True)
    test_images = read_ubyte("data/t10k-images-idx3-ubyte.gz", is_img=True)
    train_labels = read_ubyte("data/train-labels-idx1-ubyte.gz", is_img=False)
    test_labels = read_ubyte("data/t10k-labels-idx1-ubyte.gz", is_img=False)

    train_data = MnistDataLoader(train_images, train_labels)
    test_data = MnistDataLoader(test_images, test_labels)
    net = Net(train_data, test_data)

    results = []

    for image in train_data:
        result = net.forward(image["image"])
        results.append((result, image["label"]))

        print("predicted", result)
        print("label", image["label"])

    loss(results)

    plt.hist(results, bins=10)
    plt.show()

