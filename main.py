import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import gzip
import tqdm
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
        self.X = []


    def calc(self, x):
        """Function: z = Wx + b"""
        self.X = x
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

        self.layers = [
            self.L1,
            self.L2
        ]

    def forward(self, x):
        """Get prediction from nueral net"""
        x = x.reshape(784)
        x = self.L1.calc(x)
        x = np.array([sigmoid(n) for n in x])

        x = self.L2.calc(x)
        x = np.array([sigmoid(n) for n in x])

        x = softmax(x)

        return x

def loss(pred, actual, net):
    alpha = 0.1
    loss = 0
    if actual > 9:
        return 0

    ground = np.zeros(len(pred))
    ground[actual] = 1

    # Binary cross entropy loss
    # Function: −(𝑦log(𝑝)+(1−𝑦)log(1−𝑝))
    loss = -(np.dot(ground, np.log(pred.T)))

    # Update weights for each layer
    for i, z in enumerate(pred - ground):
        X = net.L2.X

        # Gradient of loss w.r.t weights vector
        dL2dw = X.T * z

        # Add weights to list to update net weights
        updatedL2 = (net.L2.weights[i] - dL2dw * alpha)
        net.L2.weights[i] = updatedL2

        for j, d in enumerate(updatedL2 - net.L2.weights[i]):
            X = net.L1.X

            # Gradient of loss w.r.t weights vector
            dL1dw = X.T * z

            # Add weights to list to update net weights
            updatedL1 = (net.L1.weights[i] - dL1dw * alpha)
            net.L1.weights[i] = updatedL1

    return (loss)

if __name__=="__main__":
    train_images = read_ubyte("data/train-images-idx3-ubyte.gz", is_img=True)
    test_images = read_ubyte("data/t10k-images-idx3-ubyte.gz", is_img=True)
    train_labels = read_ubyte("data/train-labels-idx1-ubyte.gz", is_img=False)
    test_labels = read_ubyte("data/t10k-labels-idx1-ubyte.gz", is_img=False)

    train_data = MnistDataLoader(train_images, train_labels)
    test_data = MnistDataLoader(test_images, test_labels)
    net = Net(train_data, test_data)

    train_accuracy = []
    train_loss = []
    for epoch in range(30):
        correct = 0
        running_loss = 0
        for image in tqdm.tqdm(train_data):
            result = net.forward(image["image"])

            running_loss += loss(result, image["label"], net)

            if image["label"] == np.argmax(result):
                correct += 1

        print(correct / len(train_data))
        train_accuracy.append(correct / len(train_data))
        train_loss.append(running_loss)

    plt.plot(normalize(np.array(train_accuracy)))
    plt.plot(normalize(np.array(train_loss)))
    plt.show()

