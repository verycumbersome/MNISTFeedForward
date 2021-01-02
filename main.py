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
    s = lambda k: 1 / (1 + (math.e ** (-k)))
    return(np.array([s(xi) for xi in x]))


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
    def __init__(self, train, test):
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

def delta(l, net):
    # Derivative of sigmoid(z) -> σ'(z) = σ(z)(1 - σ(z))
    fpl = net.layers[l].X * (np.ones(len(net.layers[l].X)) - net.layers[l].X)

    # Derivative of cross entropy cost function
    w = net.layers[l].weights

    if l == len(net.layers) + 1:
        print("layer" + str(l) + " prev layer: ", fpl.shape)
        print("layer" + str(l) + " weights", w.shape)
        print("layer" + str(l) + "", np.dot(fpl, w.T).shape)

        # Derivative of sigmoid(z) -> σ'(z) = σ(z)(1 - σ(z))
        dAdZ = (z) * (np.ones(len(z)) - z)

        # derivative of cross entropy cost function
        dCdA = (y - t) / dAdZ

        output = np.dot(fpl, w.T)
        return np.dot(fpl, w.T)

    print("layer" + str(l) + " prev layer: ", fpl.shape)
    print("layer" + str(l) + " weights: ", w.shape)
    print("layer" + str(l) + "", np.dot(fpl, w.T).shape)

    return np.multiply(np.dot(fpl, w.T), delta(l + 1, net))


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


    print(delta(0, net))
    exit()
    # layer = net.L2
    # y = pred
    # z = layer.layer_output
    # # Derivative of sigmoid(z) -> σ'(z) = σ(z)(1 - σ(z))
    # dAdZ = (z) * (np.ones(len(z)) - z)
    # print(dAdZ.shape)

    # # derivative of cross entropy cost function
    # dCdA = (y - t) / dAdZ

    # # Derivative of weight w.r.t z
    # dZdW = layer.X
    # print(dCdA.shape)

    # # Derivative of cost w.r.t weight
    # dCdW = (dAdZ * dCdA * dZdW)
    # layer.weights = layer.weights - dCdW * alpha




    # layer = net.L2
    # y = pred
    # for j in range(layer.out_size):
        # for k in range(layer.in_size):
            # # Derivative of sigmoid(z) -> σ'(z) = σ(z)(1 - σ(z))
            # dAdZ = (y[j]) * (1 - y[j])

            # # derivative of cross entropy cost function
            # dCdA = (y[j] - t[j]) / dAdZ

            # # Derivative of weight w.r.t z
            # dZdW = layer.layer_output[j]

            # # Derivative of cost w.r.t weight
            # dCdW = (dAdZ * dCdA * dZdW)
            # layer.weights[j][k] = layer.weights[j][k] - dCdW * alpha

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
    for epoch in range(200):
        correct = 0
        running_loss = 0
        for image in tqdm.tqdm(train_data.rand_sample(200)):
            result = net.forward(image["image"])

            running_loss += loss(result, image["label"], net)

            if image["label"] == np.argmax(result):
                correct += 1

        train_accuracy.append(correct / 200)
        train_loss.append(running_loss)

        print("Epoch:", epoch)
        print("Accuracy:", correct / 200)
        print("Loss:", running_loss)

    plt.plot(normalize(np.array(train_accuracy)))
    plt.plot(normalize(np.array(train_loss)))
    plt.show()

