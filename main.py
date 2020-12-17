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

    return(data)


train_images = read_ubyte("data/train-images-idx3-ubyte.gz", is_img=True)
test_images = read_ubyte("data/t10k-images-idx3-ubyte.gz", is_img=True)

train_labels = read_ubyte("data/train-labels-idx1-ubyte.gz", is_img=False)
test_labels = read_ubyte("data/t10k-labels-idx1-ubyte.gz", is_img=False)


class MnistDataLoader():
    def __init__(self):
        # self.data = 
        pass
    def __len__(self):
        pass
    def __getitem__(self):
        pass

class Net():
    def __init__(self):
        pass


if __name__=="__main__":
    dataloader = MnistDataLoader
    net = Net()
