
from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from skimage.transform import resize


def shuffle(xs, ys):
    indices = list(range(len(xs)))
    np.random.shuffle(indices)
    return xs[indices], ys[indices]

def mnist_resize(x):
    H, W, C = 32, 32, 3
    x = x.reshape(-1, 28, 28)
    resized_x = np.empty((len(x), H, W), dtype='float32')
    for i, img in enumerate(x):
        if i % 1000 == 0:
            print(i)
        # resize returns [0, 1]
        resized_x[i] = resize(img, (H, W), mode='reflect')

    # Retile to make RGB
    resized_x = resized_x.reshape(-1, H, W, 1)
    resized_x = np.tile(resized_x, (1, 1, 1, C))
    return resized_x

def save_mnist_32():
    (mnist_x, mnist_y), (_, _) = mnist.load_data()
    mnist_x = mnist_resize(mnist_x / 255.0)
    savemat('mnist32_train.mat', {'X': mnist_x, 'y': mnist_y})


def make_mnist_svhn_dataset(num_examples, mnist_start_prob, mnist_end_prob):
    data = loadmat('mnist32_train.mat')
    mnist_x = data['X']
    mnist_y = data['y']
    mnist_y = np.squeeze(mnist_y)
    mnist_x, mnist_y = shuffle(mnist_x, mnist_y)
    print(np.min(mnist_x), np.max(mnist_x))

    data = loadmat('svhn_train_32x32.mat')
    svhn_x = data['X']
    svhn_x = svhn_x / 255.0
    svhn_x = np.transpose(svhn_x, [3, 0, 1, 2])
    svhn_y = data['y']
    svhn_y = np.squeeze(svhn_y)
    svhn_y[(svhn_y == 10)] = 0
    svhn_x, svhn_y = shuffle(svhn_x, svhn_y)
    print(svhn_x.shape, svhn_y.shape)
    print(np.min(svhn_x), np.max(svhn_x))

    delta = float(mnist_end_prob - mnist_start_prob) / (num_examples - 1)
    mnist_probs = np.array([mnist_start_prob + delta * i for i in range(num_examples)])
    # assert((np.all(mnist_end_prob >= mnist_probs) and np.all(mnist_probs >= mnist_start_prob)) or
    #      (np.all(mnist_start_prob >= mnist_probs) and np.all(mnist_probs >= mnist_end_prob)))
    domains = np.random.binomial(n=1, p=mnist_probs)
    assert(domains.shape == (num_examples,))
    mnist_indices = np.arange(num_examples)[domains == 1]
    svhn_indices = np.arange(num_examples)[domains == 0]
    print(svhn_x.shape, mnist_x.shape)
    assert(svhn_x.shape[1:] == mnist_x.shape[1:])
    print(mnist_indices[:10], svhn_indices[:10], svhn_indices[-10:])
    xs = np.empty((num_examples,) + tuple(svhn_x.shape[1:]), dtype='float32')
    ys = np.empty((num_examples,), dtype='int32')
    xs[mnist_indices] = mnist_x[:mnist_indices.size]
    xs[svhn_indices] = svhn_x[:svhn_indices.size]
    ys[mnist_indices] = mnist_y[:mnist_indices.size]
    ys[svhn_indices] = svhn_y[:svhn_indices.size]
    return xs, ys



save_mnist_32()
# xs, ys = make_mnist_svhn_dataset(10000, 0.9, 0.1)
# print(xs.shape, ys.shape)
# ex_0 = xs[ys == 0][0]
# plt.imshow(ex_0)
# plt.show()
# ex_0 = xs[ys == 0][-1]
# plt.imshow(ex_0)
# plt.show()


    # Read and process MNIST images
    # Read and process SVHN images
    # Shuffle MNIST and SVHN images
    # First generate an array of datasets
        # Interpolate between start and end prob
        # Use that to pick and index from a Bernoulli
    # Then select the images (could do some fancy indexing, or just do it manually, append images)
    # Should be fast enough, did that in rotated dataset
    # Return this

# data = loadmat('svhn_train_32x32.mat')
# Xs = data['X']
# Xs = np.transpose(Xs, [3, 0, 1, 2])
# Ys = data['y']
# Ys = np.squeeze(Ys)
# print(Xs.shape, Ys.shape)
# print(np.min(Ys), np.max(Ys))
# Ys[(Ys == 10)] = 0
# print(np.min(Ys), np.max(Ys))
# print(np.min(Xs), np.max(Xs))

# Resize MNIST images to make them colored. Just add extra channels.
# Need to check min and max for MNIST and SVHN, preprocess them as needed
# Could add instance normalization as well, if we think that helps.
# (train_x, train_y), (_, _) = mnist.load_data()
# print(np.min(train_x), np.max(train_x))
# train_x = np.tile(np.expand_dims(train_x, axis=-1), (1, 1, 1, 3))
# print(np.min(train_y), np.max(train_y))
# print(train_y.shape)
# ex_0 = train_x[train_y == 0][0]
# plt.imshow(ex_0)
# plt.show()
# print(train_x.shape)

# X10s = Xs[y10s]
# print(X10s.shape)
# plt.imshow(X10s[0])
# plt.show()