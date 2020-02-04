
import collections
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
import scipy.io
from scipy import ndimage
from scipy.stats import ortho_group
import sklearn.preprocessing
import pickle
import utils

Dataset = collections.namedtuple('Dataset',
    'get_data n_src_train n_src_valid n_target_unsup n_target_val n_target_test target_end '
    'n_classes input_shape')

SplitData = collections.namedtuple('SplitData',
    ('src_train_x src_val_x src_train_y src_val_y target_unsup_x target_val_x final_target_test_x '
     'debug_target_unsup_y target_val_y final_target_test_y inter_x inter_y'))


image_options = {
    'batch_size': 100,
    'class_mode': 'binary',
    'color_mode': 'grayscale',
}


def split_sizes(array, sizes):
    indices = np.cumsum(sizes)
    return np.split(array, indices)


def shuffle(xs, ys):
    indices = list(range(len(xs)))
    np.random.shuffle(indices)
    return xs[indices], ys[indices]


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


# Gaussian dataset.

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
    new_sigmas = sigmas
    if shape_len == 1:
        c = np.expand_dims(np.expand_dims(sigmas, axis=-1), axis=-1)
        d = means.shape[1]
        new_sigmas = c * np.eye(d)
        assert(new_sigmas.shape == (sigmas.shape[0], d, d))
    return new_sigmas


def get_gaussian_at_alpha(source_means, source_sigmas, target_means, target_sigmas, alpha):
    num_classes = source_means.shape[0]
    class_prob = 1.0 / num_classes
    y = np.argmax(np.random.multinomial(1, [class_prob] * num_classes))
    mean = source_means[y] * (1 - alpha) + target_means[y] * alpha
    sigma = source_sigmas[y] * (1 - alpha) + target_sigmas[y] * alpha
    x = np.random.multivariate_normal(mean, sigma)
    return x, y


def sample_gaussian_alpha(source_means, source_sigmas, target_means, target_sigmas,
                          alpha_low, alpha_high, N):
    source_means = shape_means(source_means)
    target_means = shape_means(target_means)
    source_sigmas = shape_sigmas(source_sigmas, source_means)
    target_sigmas = shape_sigmas(target_sigmas, target_means)
    xs, ys = [], []
    for i in range(N):
        if alpha_low == alpha_high:
            alpha = alpha_low
        else:
            alpha = np.random.uniform(low=alpha_low, high=alpha_high)
        x, y = get_gaussian_at_alpha(
            source_means, source_sigmas, target_means, target_sigmas, alpha)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def continual_gaussian_alpha(source_means, source_sigmas, target_means, target_sigmas,
                             alpha_low, alpha_high, N):
    source_means = shape_means(source_means)
    target_means = shape_means(target_means)
    source_sigmas = shape_sigmas(source_sigmas, source_means)
    target_sigmas = shape_sigmas(target_sigmas, target_means)
    xs, ys = [], []
    for i in range(N):
        alpha = float(alpha_high - alpha_low) / N * i + alpha_low
        x, y = get_gaussian_at_alpha(
            source_means, source_sigmas, target_means, target_sigmas, alpha)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def make_moving_gaussian_data(
    source_means, source_sigmas, target_means, target_sigmas,
    source_alphas, inter_alphas, target_alphas,
    n_src_tr, n_src_val, n_inter, n_trg_val, n_trg_tst):
    src_tr_x, src_tr_y = sample_gaussian_alpha(
        source_means, source_sigmas, target_means, target_sigmas,
        source_alphas[0], source_alphas[1], N=n_src_tr)
    src_val_x, src_val_y = sample_gaussian_alpha(
        source_means, source_sigmas, target_means, target_sigmas,
        source_alphas[0], source_alphas[1], N=n_src_val)
    inter_x, inter_y = continual_gaussian_alpha(
        source_means, source_sigmas, target_means, target_sigmas,
        inter_alphas[0], inter_alphas[1], N=n_inter)
    dir_inter_x, dir_inter_y = sample_gaussian_alpha(
        source_means, source_sigmas, target_means, target_sigmas,
        target_alphas[0], target_alphas[1], N=n_inter)
    trg_val_x, trg_val_y = sample_gaussian_alpha(
        source_means, source_sigmas, target_means, target_sigmas,
        target_alphas[0], target_alphas[1], N=n_trg_val)
    trg_test_x, trg_test_y = sample_gaussian_alpha(
        source_means, source_sigmas, target_means, target_sigmas,
        target_alphas[0], target_alphas[1], N=n_trg_tst)
    return (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y,
            dir_inter_x, dir_inter_y, trg_val_x, trg_val_y, trg_test_x, trg_test_y)


def make_high_d_gaussian_data(
    d, min_var, max_var, source_alphas, inter_alphas, target_alphas,
    n_src_tr, n_src_val, n_inter, n_trg_val, n_trg_tst):
    assert(min_var > 0)
    means, var_list = [], []
    for i in range(4):
        means.append(np.random.multivariate_normal(np.zeros(d), np.eye(d)))
        means[i] = means[i] / np.linalg.norm(means[i])
        # Generate diagonal.
        diag = np.diag(np.random.uniform(min_var, max_var, size=d))
        rot = ortho_group.rvs(d)
        var = np.matmul(rot, np.matmul(diag, np.linalg.inv(rot)))
        var_list.append(var)
    return make_moving_gaussian_data(
        source_means=[means[0], means[1]], source_sigmas=[var_list[0], var_list[1]],
        target_means=[means[2], means[3]], target_sigmas=[var_list[2], var_list[3]],
        source_alphas=source_alphas, inter_alphas=inter_alphas, target_alphas=target_alphas,
        n_src_tr=n_src_tr, n_src_val=n_src_val, n_inter=n_inter,
        n_trg_val=n_trg_val, n_trg_tst=n_trg_tst)


def make_moving_gaussians(source_means, source_sigmas, target_means, target_sigmas, steps):
    source_means = shape_means(source_means)
    target_means = shape_means(target_means)
    source_sigmas = shape_sigmas(source_sigmas, source_means)
    target_sigmas = shape_sigmas(target_sigmas, target_means)
    for i in range(steps):
        alpha = float(i) / (steps - 1)
        mean = source_means[y] * (1 - alpha) + target_means[y] * alpha
        sigma = source_sigmas[y] * (1 - alpha) + target_sigmas[y] * alpha
        x, y = get_gaussian_at_alpha()
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


# MNIST datasets.

def get_preprocessed_mnist():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x, test_x = train_x / 255.0, test_x / 255.0
    train_x, train_y = shuffle(train_x, train_y)
    train_x = np.expand_dims(np.array(train_x), axis=-1)
    test_x = np.expand_dims(np.array(test_x), axis=-1)
    return (train_x, train_y), (test_x, test_y)


def sample_rotate_images(xs, start_angle, end_angle):
    new_xs = []
    num_points = xs.shape[0]
    for i in range(num_points):
        if start_angle == end_angle:
            angle = start_angle
        else:
            angle = np.random.uniform(low=start_angle, high=end_angle)
        img = ndimage.rotate(xs[i], angle, reshape=False)
        new_xs.append(img)
    return np.array(new_xs)


def continually_rotate_images(xs, start_angle, end_angle):
    new_xs = []
    num_points = xs.shape[0]
    for i in range(num_points):
        angle = float(end_angle - start_angle) / num_points * i + start_angle
        img = ndimage.rotate(xs[i], angle, reshape=False)
        new_xs.append(img)
    return np.array(new_xs)


def _transition_rotation_dataset(train_x, train_y, test_x, test_y,
                                 source_angles, target_angles, inter_func,
                                 src_train_end, src_val_end, inter_end, target_end):
    assert(target_end <= train_x.shape[0])
    assert(train_x.shape[0] == train_y.shape[0])
    src_tr_x, src_tr_y = train_x[:src_train_end], train_y[:src_train_end]
    src_tr_x = sample_rotate_images(src_tr_x, source_angles[0], source_angles[1])
    src_val_x, src_val_y = train_x[src_train_end:src_val_end], train_y[src_train_end:src_val_end]
    src_val_x = sample_rotate_images(src_val_x, source_angles[0], source_angles[1])
    tmp_inter_x, inter_y = train_x[src_val_end:inter_end], train_y[src_val_end:inter_end]
    inter_x = inter_func(tmp_inter_x)
    dir_inter_x = sample_rotate_images(tmp_inter_x, target_angles[0], target_angles[1])
    dir_inter_y = np.array(inter_y)
    assert(inter_x.shape == dir_inter_x.shape)
    trg_val_x, trg_val_y = train_x[inter_end:target_end], train_y[inter_end:target_end]
    trg_val_x = sample_rotate_images(trg_val_x, target_angles[0], target_angles[1])
    trg_test_x, trg_test_y = test_x, test_y
    trg_test_x = sample_rotate_images(trg_test_x, target_angles[0], target_angles[1])
    return (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y,
            dir_inter_x, dir_inter_y, trg_val_x, trg_val_y, trg_test_x, trg_test_y)


def dial_rotation_proportions(xs, source_angles, target_angles):
    N = xs.shape[0]
    new_xs = []
    rotate_ps = np.arange(N) / float(N - 1)
    is_target = np.random.binomial(n=1, p=rotate_ps)
    assert(is_target.shape == (N,))
    for i in range(N):
        if is_target[i]:
            angle = np.random.uniform(low=target_angles[0], high=target_angles[1])
        else:
            angle = np.random.uniform(low=source_angles[0], high=source_angles[1])
        cur_x = ndimage.rotate(xs[i], angle, reshape=False)
        new_xs.append(cur_x)
    return np.array(new_xs)


def dial_proportions_rotated_dataset(train_x, train_y, test_x, test_y,
                                     source_angles, target_angles,
                                     src_train_end, src_val_end, inter_end, target_end):
    inter_func = lambda x: dial_rotation_proportions(
        x, source_angles, target_angles)
    return _transition_rotation_dataset(
        train_x, train_y, test_x, test_y, source_angles, target_angles,
        inter_func, src_train_end, src_val_end, inter_end, target_end)


def make_rotated_dataset(train_x, train_y, test_x, test_y,
                         source_angles, inter_angles, target_angles,
                         src_train_end, src_val_end, inter_end, target_end):
    inter_func = lambda x: continually_rotate_images(x, inter_angles[0], inter_angles[1])
    return _transition_rotation_dataset(
        train_x, train_y, test_x, test_y, source_angles, target_angles,
        inter_func, src_train_end, src_val_end, inter_end, target_end)


def make_population_rotated_dataset(xs, ys, delta_angle, num_angles):
    images, labels = [], []
    for i in range(num_angles):
        cur_angle = i * delta_angle
        cur_images = sample_rotate_images(xs, cur_angle, cur_angle)
        images.append(cur_images)
        labels.append(ys)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    assert images.shape[1:] == xs.shape[1:]
    assert labels.shape[1:] == ys.shape[1:]
    return images, labels


def make_rotated_dataset_continuous(dataset, start_angle, end_angle, num_points):
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


# Portraits dataset.

def save_data(data_dir='dataset_32x32', save_file='dataset_32x32.mat', target_size=(32, 32)):
    Xs, Ys = [], []
    datagen = ImageDataGenerator(rescale=1./255)
    data_generator = datagen.flow_from_directory(
        data_dir, shuffle=False, target_size=target_size, **image_options)
    while True:
        next_x, next_y = data_generator.next()
        Xs.append(next_x)
        Ys.append(next_y)
        if data_generator.batch_index == 0:
            break
    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
    filenames = [f[2:] for f in data_generator.filenames]
    assert(len(set(filenames)) == len(filenames))
    filenames_idx = list(zip(filenames, range(len(filenames))))
    filenames_idx = [(f, i) for f, i in zip(filenames, range(len(filenames)))]
                     # if f[5:8] == 'Cal' or f[5:8] == 'cal']
    indices = [i for f, i in sorted(filenames_idx)]
    genders = np.array([f[:1] for f in data_generator.filenames])[indices]
    binary_genders = (genders == 'F')
    pickle.dump(binary_genders, open('portraits_gender_stats', "wb"))
    print("computed gender stats")
    # gender_stats = utils.rolling_average(binary_genders, 500)
    # print(filenames)
    # sort_indices = np.argsort(filenames)
    # We need to sort only by year, and not have correlation with state.
    # print state stats? print gender stats? print school stats?
    # E.g. if this changes a lot by year, then we might want to do some grouping.
    # Maybe print out number per year, and then we can decide on a grouping? Or algorithmically decide?
    Xs = Xs[indices]
    Ys = Ys[indices]
    scipy.io.savemat('./' + save_file, mdict={'Xs': Xs, 'Ys': Ys})


def load_portraits_data(load_file='dataset_32x32.mat'):
    data = scipy.io.loadmat('./' + load_file)
    return data['Xs'], data['Ys'][0]


def make_portraits_data(n_src_tr, n_src_val, n_inter, n_target_unsup, n_trg_val, n_trg_tst,
                        load_file='dataset_32x32.mat'):
    xs, ys = load_portraits_data(load_file)
    src_end = n_src_tr + n_src_val
    inter_end = src_end + n_inter
    trg_end = inter_end + n_trg_val + n_trg_tst
    src_x, src_y = shuffle(xs[:src_end], ys[:src_end])
    trg_x, trg_y = shuffle(xs[inter_end:trg_end], ys[inter_end:trg_end])
    [src_tr_x, src_val_x] = split_sizes(src_x, [n_src_tr])
    [src_tr_y, src_val_y] = split_sizes(src_y, [n_src_tr])
    [trg_val_x, trg_test_x] = split_sizes(trg_x, [n_trg_val])
    [trg_val_y, trg_test_y] = split_sizes(trg_y, [n_trg_val])
    inter_x, inter_y = xs[src_end:inter_end], ys[src_end:inter_end]
    dir_inter_x, dir_inter_y = inter_x[-n_target_unsup:], inter_y[-n_target_unsup:]
    return (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y,
            dir_inter_x, dir_inter_y, trg_val_x, trg_val_y, trg_test_x, trg_test_y)


def rotated_mnist_60_data_func():
    (train_x, train_y), (test_x, test_y) = get_preprocessed_mnist()
    return make_rotated_dataset(
        train_x, train_y, test_x, test_y, [0.0, 5.0], [5.0, 60.0], [55.0, 60.0],
        5000, 6000, 48000, 50000)


def rotated_mnist_60_dialing_ratios_data_func():
    (train_x, train_y), (test_x, test_y) = get_preprocessed_mnist()
    return dial_proportions_rotated_dataset(
        train_x, train_y, test_x, test_y, [0.0, 5.0], [55.0, 60.0],
        5000, 6000, 48000, 50000)


def portraits_data_func():
    return make_portraits_data(1000, 1000, 14000, 2000, 1000, 1000)


def portraits_64_data_func():
    return make_portraits_data(1000, 1000, 14000, 2000, 1000, 1000, load_file='dataset_64x64.mat')


def gaussian_data_func(d):
    return make_high_d_gaussian_data(
        d=d, min_var=0.05, max_var=0.1,
        source_alphas=[0.0, 0.0], inter_alphas=[0.0, 1.0], target_alphas=[1.0, 1.0],
        n_src_tr=500, n_src_val=1000, n_inter=5000, n_trg_val=1000, n_trg_tst=1000)



