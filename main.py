import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import tqdm
from dataclasses import dataclass


def read_ubyte(filepath, is_img = True, image_size = 28, num_samples = 10000):
    f = gzip.open(filepath, "r")
    if (is_img):
        num_samples = 3
        buf = f.read(image_size * image_size * num_samples)
        data = np.frombuffer(buf, dtype = np.uint8).astype(np.float32)
        data = data.reshape(num_samples, image_size, image_size)

        # plt.imshow(data[2], cmap="gray")
        # plt.show()
        # exit()


    else:
        buf = f.read(num_samples)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)


    return(data.squeeze())


def ReLU(x):
    return(x if x > 0 else 0)


def sigmoid(x):
    s = lambda k: 1 / (1 + (math.e ** (-k)))
    return(np.array([s(xi) for xi in x]))


def sigprime(x):
    sA = sigmoid(x) * (np.ones(len(x)) - sigmoid(x))
    return(sA)


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

    def rand_sample(self, n):
        r = random.randint(0, len(self.images))

        output = []
        for i in range(r, r + n):
            idx = (i % len(self.images))
            output.append({
                "image":self.images[idx],
                "label":int(self.labels[idx]),
            })
        return output


@dataclass
class LinearLayer():
    in_size: int
    out_size: int

    def __post_init__(self):
        # For each node in output layer, generate empty weights and biases
        # self.weights = np.random.uniform(0, 1, size =[self.out_size, self.in_size])
        self.weights = np.random.randn(self.out_size, self.in_size) * \
                np.sqrt(2 / self.in_size)
        self.biases = np.random.uniform(0, 1, self.out_size)
        self.X = []

    def calc(self, x):
        """Function: z = Wx + b"""
        self.X = x

        self.z = np.dot(self.weights, x) + self.biases
        self.layer_output = sigmoid(self.z)

        return(self.layer_output)


class Net():
    def __init__(self):
        self.L1 = LinearLayer(784, 50)
        self.L2 = LinearLayer(50, 10)

        self.layers = [
            self.L1,
            self.L2,
        ]

    def forward(self, x):
        """Get prediction from nueral net"""
        x = x.reshape(784)

        x = self.L1.calc(x)
        x = self.L2.calc(x)

        # x = softmax(x)

        return x

    def backprop():
        pass


def delta(l, t, net):
    # Derivative of sigmoid(z) -> σ'(z) = σ(z)(1 - σ(z))
    dA = sigprime(net.layers[l].z)

    if l == len(net.layers) - 1:
        # derivative of cross entropy cost function
        y = net.layers[l].layer_output
        dCdA = -1 * ((t / y) + ((1 - t) / (1 - y)))

        return np.multiply(dCdA, dA)

    # Get the weights at the next layer
    w = net.layers[l + 1].weights

    return np.multiply(np.dot(w.T, delta(l + 1, t, net)), dA)


def loss(pred, actual, net):
    alpha = 0.1
    loss = 0
    if actual > 9:
        return 0

    t = np.zeros(len(pred))
    t[actual] = 1

    # Binary cross entropy loss
    # Function: −(𝑦log(𝑝)+(1−𝑦)log(1−𝑝))
    # for i in range(len(t)):
        # loss -= t[i] * math.log(pred[i]) + (1 - t[i]) * math.log(1 - pred[i])

    D1 = delta(0, t, net)
    D2 = delta(1, t, net)

    for i in range(50):
        for j in range(784):
            print(net.L1.weights[i][j])
    net.L1.weights -= D1 * net.L2.X
    # net.L1.weights -= D1 * net.L1.X

    net.L1.biases -= D1
    net.L2.biases -= D2

    return (loss)


if __name__=="__main__":
    data_train = pd.read_csv("data/train.csv")

    train_labels = np.array(data_train.iloc[:, 0])
    train_images = np.array(data_train.iloc[:, 1:]).reshape(42000, 28, 28)

    train_data = MnistDataLoader(train_images, train_labels)
    net = Net()

    train_accuracy = []
    train_loss = []
    for epoch in range(200):
        correct = 0
        counter = 0
        running_loss = 0
        for image in tqdm.tqdm(train_data.rand_sample(200)):
            result = net.forward(image["image"])

            running_loss += loss(result, image["label"], net)

            if image["label"] == np.argmax(result):
                correct += 1

            counter += 1

        train_accuracy.append(correct / counter)
        train_loss.append(running_loss)

        print("Epoch:", epoch)
        print("Accuracy:", correct / counter)
        print("Loss:", running_loss)

    plt.plot(normalize(np.array(train_accuracy)))
    plt.plot(normalize(np.array(train_loss)))
    plt.show()

