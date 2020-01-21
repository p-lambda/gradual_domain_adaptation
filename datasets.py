
import collections
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
import scipy.io
from scipy import ndimage
import sklearn.preprocessing


Dataset = collections.namedtuple('Dataset',
    'get_data n_src_train n_src_valid n_target_unsup n_target_val n_target_test target_end '
    'n_classes input_shape')

SplitData = collections.namedtuple('SplitData',
    ('src_train_x src_val_x src_train_y src_val_y target_unsup_x target_val_x final_target_test_x '
     'debug_target_unsup_y target_val_y final_target_test_y inter_x inter_y'))


def save_data(dir='dataset_32x32', save_file='dataset_32x32.mat'):
    image_options = {
        'target_size': (32, 32),
        'batch_size': 100,
        'class_mode': 'binary',
        'color_mode': 'grayscale',
    }
    Xs, Ys = [], []
    datagen = ImageDataGenerator(rescale=1./255)
    data_generator = datagen.flow_from_directory(
        'dataset_32x32/', shuffle=False, **image_options)
    while True:
        next_x, next_y = data_generator.next()
        Xs.append(next_x)
        Ys.append(next_y)
        if data_generator.batch_index == 0:
            break
    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
    filenames = [f[2:] for f in data_generator.filenames]
    filenames_idx = list(zip(filenames, range(len(filenames))))
    filenames_idx = [(f, i) for f, i in zip(filenames, range(len(filenames)))]
                     # if f[5:8] == 'Cal' or f[5:8] == 'cal']
    indices = [i for f, i in sorted(filenames_idx)]
    # print(filenames)
    # sort_indices = np.argsort(filenames)
    # We need to sort only by year, and not have correlation with state.
    # print state stats? print gender stats? print school stats?
    # E.g. if this changes a lot by year, then we might want to do some grouping.
    # Maybe print out number per year, and then we can decide on a grouping? Or algorithmically decide?
    Xs = Xs[indices]
    Ys = Ys[indices]
    scipy.io.savemat('./' + save_file, mdict={'Xs': Xs, 'Ys': Ys})


def load_faces_data(load_file='dataset_32x32.mat'):
    data = scipy.io.loadmat('./' + load_file)
    return data['Xs'], data['Ys'][0]


def split_sizes(array, sizes):
    indices = np.cumsum(sizes)
    return np.split(array, indices)


def shuffle(xs, ys):
    indices = list(range(len(xs)))
    np.random.shuffle(indices)
    return xs[indices], ys[indices]


# Synthetic Datasets.

def make_mnist_svhn_dataset(num_examples, mnist_start_prob, mnist_end_prob):
    data = scipy.io.loadmat('mnist32_train.mat')
    mnist_x = data['X']
    mnist_y = data['y']
    mnist_y = np.squeeze(mnist_y)
    mnist_x, mnist_y = shuffle(mnist_x, mnist_y)

    data = scipy.io.loadmat('svhn_train_32x32.mat')
    svhn_x = data['X']
    svhn_x = svhn_x / 255.0
    svhn_x = np.transpose(svhn_x, [3, 0, 1, 2])
    svhn_y = data['y']
    svhn_y = np.squeeze(svhn_y)
    svhn_y[(svhn_y == 10)] = 0
    svhn_x, svhn_y = shuffle(svhn_x, svhn_y)

    delta = float(mnist_end_prob - mnist_start_prob) / (num_examples - 1)
    mnist_probs = np.array([mnist_start_prob + delta * i for i in range(num_examples)])
    # assert((np.all(mnist_end_prob >= mnist_probs) and np.all(mnist_probs >= mnist_start_prob)) or
    #      (np.all(mnist_start_prob >= mnist_probs) and np.all(mnist_probs >= mnist_end_prob)))
    domains = np.random.binomial(n=1, p=mnist_probs)
    assert(domains.shape == (num_examples,))
    mnist_indices = np.arange(num_examples)[domains == 1]
    svhn_indices = np.arange(num_examples)[domains == 0]
    assert(svhn_x.shape[1:] == mnist_x.shape[1:])
    xs = np.empty((num_examples,) + tuple(svhn_x.shape[1:]), dtype='float32')
    ys = np.empty((num_examples,), dtype='int32')
    xs[mnist_indices] = mnist_x[:mnist_indices.size]
    xs[svhn_indices] = svhn_x[:svhn_indices.size]
    ys[mnist_indices] = mnist_y[:mnist_indices.size]
    ys[svhn_indices] = svhn_y[:svhn_indices.size]
    return xs, ys


def make_rotated_dataset(dataset, start_angle, end_angle, num_points):
    images, labels = [], []
    (train_x, train_y), (_, _) = dataset.load_data()
    train_x, train_y = shuffle(train_x, train_y)
    train_x = train_x / 255.0
    assert(num_points < train_x.shape[0])
    indices = np.random.choice(train_x.shape[0], size=num_points, replace=False)
    for i in range(num_points):
        angle = float(end_angle - start_angle) / num_points * i + start_angle
        idx = indices[i]
        img = ndimage.rotate(train_x[idx], angle, reshape=False)
        images.append(img)
        labels.append(train_y[idx])
    return np.array(images), np.array(labels)


def make_rotated_mnist(start_angle, end_angle, num_points, normalize=False):
    Xs, Ys = make_rotated_dataset(mnist, start_angle, end_angle, num_points)
    if normalize:
        Xs = np.reshape(Xs, (Xs.shape[0], -1))
        old_mean = np.mean(Xs)
        Xs = sklearn.preprocessing.normalize(Xs, norm='l2')
        new_mean = np.mean(Xs)
        Xs = Xs * (old_mean / new_mean)
    return np.expand_dims(np.array(Xs), axis=-1), Ys


def make_rotated_cifar10(start_angle, end_angle, num_points):
    return make_rotated_dataset(cifar10, start_angle, end_angle, num_points)   


def make_mnist():
    (train_x, train_y), (_, _) = mnist.load_data()
    train_x = train_x / 255.0
    return np.expand_dims(train_x, axis=-1), train_y


def make_moving_gaussians(source_means, source_sigmas, target_means, target_sigmas, steps):
    def shape_means(means):
        means = np.array(means)
        if len(means.shape) == 1:
            means = np.expand_dims(means, axis=-1)
        else:
            assert(len(means.shape) == 2)
        return means
    def shape_sigmas(sigmas, means):
        sigmas = np.array(sigmas)
        shape_len = len(sigmas.shape)
        assert(shape_len == 1 or shape_len == 3)
        if shape_len == 1:
            c = np.expand_dims(np.expand_dims(sigmas, axis=-1), axis=-1)
            d = means.shape[1]
            new_sigmas = c * np.eye(d)
            assert(new_sigmas.shape == (sigmas.shape[0], d, d))
        return new_sigmas
    source_means = shape_means(source_means)
    target_means = shape_means(target_means)
    source_sigmas = shape_sigmas(source_sigmas, source_means)
    target_sigmas = shape_sigmas(target_sigmas, target_means)
    num_classes = source_means.shape[0]
    class_prob = 1.0 / num_classes
    xs = []
    ys = []
    for i in range(steps):
        y = np.argmax(np.random.multinomial(1, [class_prob] * num_classes))
        alpha = float(i) / (steps - 1)
        mean = source_means[y] * (1 - alpha) + target_means[y] * alpha
        sigma = source_sigmas[y] * (1 - alpha) + target_sigmas[y] * alpha
        x = np.random.multivariate_normal(mean, sigma)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def high_d_gaussians(d, var, n):
    # Choose random direction.
    v = np.random.multivariate_normal(np.zeros(d), np.eye(d))
    v = v / np.linalg.norm(v)
    # Choose random perpendicular direction.
    perp = np.random.multivariate_normal(np.zeros(d), np.eye(d))
    perp = perp - np.dot(perp, v) * v
    perp = perp / np.linalg.norm(perp)
    assert(abs(np.dot(perp, v)) < 1e-8)
    assert(abs(np.linalg.norm(v) - 1) < 1e-8)
    assert(abs(np.linalg.norm(perp) - 1) < 1e-8)
    s_a = 2 * perp - v
    s_b = -2 * perp + v
    t_a = -2 * perp - v
    t_b = 2 * perp + v
    return lambda: make_moving_gaussians([s_a, s_b], [var, var], [t_a, t_b], [var, var], n)


def get_split_data(dataset):
    Xs, Ys = dataset.get_data()
    n_src = dataset.n_src_train + dataset.n_src_valid
    n_target = dataset.n_target_unsup + dataset.n_target_val + dataset.n_target_test
    src_x, src_y = shuffle(Xs[:n_src], Ys[:n_src])
    target_x, target_y = shuffle(
        Xs[dataset.target_end-n_target:dataset.target_end],
        Ys[dataset.target_end-n_target:dataset.target_end])
    [src_train_x, src_val_x] = split_sizes(src_x, [dataset.n_src_train])
    [src_train_y, src_val_y] = split_sizes(src_y, [dataset.n_src_train])
    [target_unsup_x, target_val_x, final_target_test_x] = split_sizes(
        target_x, [dataset.n_target_unsup, dataset.n_target_val])
    [debug_target_unsup_y, target_val_y, final_target_test_y] = split_sizes(
        target_y, [dataset.n_target_unsup, dataset.n_target_val])
    inter_x, inter_y = Xs[n_src:dataset.target_end-n_target], Ys[n_src:dataset.target_end-n_target]
    return SplitData(
        src_train_x=src_train_x,
        src_val_x=src_val_x,
        src_train_y=src_train_y,
        src_val_y=src_val_y,
        target_unsup_x=target_unsup_x,
        target_val_x=target_val_x,
        final_target_test_x=final_target_test_x,
        debug_target_unsup_y=debug_target_unsup_y,
        target_val_y=target_val_y,
        final_target_test_y=final_target_test_y,
        inter_x=inter_x,
        inter_y=inter_y,
    )
