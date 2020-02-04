import scipy.io as sio
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy.io
from tensorflow.keras.datasets import mnist
from scipy import ndimage
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import regularizers
import collections


image_options = {
    'target_size': (32, 32),
    'batch_size': 100,
    'class_mode': 'binary',
    'color_mode': 'grayscale',
}

np.random.seed(0)
tf.compat.v1.set_random_seed(0)

def linear_model(num_labels, input_shape, l2_reg=0.02):
    print(l2_reg)
    linear_model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(num_labels, activation=None, name='out',
        kernel_regularizer=regularizers.l2(l2_reg))
        ])
    return linear_model


def simple_conv_model(num_labels, hidden_nodes=32, input_shape=(28,28,1)):
    return keras.models.Sequential([
    keras.layers.Conv2D(hidden_nodes, (5,5), (2,2), activation=tf.nn.relu,
                           padding='same', input_shape=input_shape),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2,2), activation=tf.nn.relu,
                           padding='same'),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2,2), activation=tf.nn.relu,
                           padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(name='after_flatten'),
    # keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
    ])


def papernot_model(num_labels, input_shape=(28,28,1)):
    papernot_conv_model = keras.models.Sequential([
     keras.layers.Conv2D(64, (8, 8), (2,2), activation=tf.nn.relu,
                            padding='same', input_shape=input_shape),
     keras.layers.Conv2D(128, (6,6), (2,2), activation=tf.nn.relu,
                            padding='valid'),
     keras.layers.Conv2D(128, (5,5), (1,1), activation=tf.nn.relu,
                            padding='valid'),
     keras.layers.Flatten(name='after_flatten'),
     keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
    ])
    return papernot_conv_model


def save_data(dir='dataset_32x32', save_file='dataset_32x32.mat'):
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


def make_rotated_mnist(start_angle, end_angle, num_points):
    images, labels = [], []
    (train_x, train_y), (_, _) = mnist.load_data()
    train_x = train_x / 255.0
    for i in range(num_points):
        angle = float(end_angle - start_angle) / num_points * i + start_angle
        idx = np.random.choice(train_x.shape[0], 1)[0]
        img = ndimage.rotate(train_x[idx], angle, reshape=False)
        images.append(img)
        labels.append(train_y[idx])
    Xs = np.expand_dims(np.array(images), axis=-1)
    return Xs, np.array(labels)


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


# Dataset-configs

Dataset = collections.namedtuple('Dataset',
    'get_data n_src_train n_src_valid n_target_unsup n_target_val n_target_test target_end '
    'n_classes input_shape')

SplitData = collections.namedtuple('SplitData',
    ('src_train_x src_val_x src_train_y src_val_y target_unsup_x target_val_x final_target_test_x '
     'debug_target_unsup_y target_val_y final_target_test_y inter_x inter_y'))

faces = Dataset(
    get_data = lambda: load_faces_data(),
    n_src_train = 1000,
    n_src_valid = 1000,
    n_target_unsup = 2000,
    n_target_val = 1000,
    n_target_test = 1000,
    target_end=20000,
    n_classes=2,
    input_shape=(32,32,1),
)

rot_mnist_0_25_15000 = Dataset(
    get_data = lambda: make_rotated_mnist(0.0, 35.0, 29000),
    n_src_train = 2000,
    n_src_valid = 1000,
    n_target_unsup = 2000,
    n_target_val = 1000,
    n_target_test = 1000,
    target_end=29000,
    n_classes=10,
    input_shape=(28,28,1),
)

gauss_2D_1K_high_noise = Dataset(
    get_data = lambda: make_moving_gaussians(
        [[2, -1], [-2, 1]], [0.8, 0.8], [[-2, -1], [2, 1]], [0.8, 0.8], 10000),
    n_src_train = 500,
    n_src_valid = 500,
    n_target_unsup = 1000,
    n_target_val = 500,
    n_target_test = 500,
    target_end=10000,
    n_classes=2,
    input_shape=(2,),
)


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


def sparse_categorical_hinge(num_classes):
    def loss(y_true,y_pred):
        y_true = tf.reduce_mean(y_true, axis=1)
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=num_classes)
        return losses.categorical_hinge(y_true, y_pred)
    return loss


def run_model_on_train(model, split_data, train_x, train_y, epochs=1000):
    model.compile(optimizer='adam',
                  loss=[sparse_categorical_hinge(2)],
                  metrics=[metrics.sparse_categorical_accuracy])
    model.fit(train_x, train_y, epochs=epochs, verbose=False)
    print("Source accuracy:")
    model.evaluate(split_data.src_val_x, split_data.src_val_y)
    print("Target accuracy:")
    model.evaluate(split_data.target_val_x, split_data.target_val_y)


def run_model(model, split_data, epochs=1000):
    run_model_on_train(model, split_data, split_data.src_train_x,
        split_data.src_train_y, epochs=epochs)


def rolling_accuracy(model, split_data, interval=1000, verbose=False):
    print("rolling accuracies")
    upper_idx = int(split_data.inter_x.shape[0] / interval)
    accs = []
    for i in range(upper_idx):
        cur_xs = split_data.inter_x[interval*i:interval*(i+1)]
        cur_ys = split_data.inter_y[interval*i:interval*(i+1)]
        _, a = model.evaluate(cur_xs, cur_ys, verbose=verbose)
        accs.append(a)
    return accs


def plot(points, labels, model, index):
    plt.cla()
    # Plot data point colored by labels.
    class_vals = [[], []]
    for i in range(2):
        class_vals[i] = [p for p, l in zip(points, labels) if l == i]
        plt.scatter([p[0] for p in class_vals[i]], [p[1] for p in class_vals[i]])
    # Plot the classifier line.
    weights = model.get_weights()[0]
    biases = model.get_weights()[1]
    x_coeff = weights[0][0] - weights[0][1]
    y_coeff = weights[1][0] - weights[1][1]
    bias = biases[0] - biases[1]
    line_func = lambda x: (-bias - x_coeff * x) / y_coeff
    xs = [p[0] for p in points]
    ys = [line_func(x) for x in xs]
    plt.plot(xs, ys)
    plt.tight_layout()
    plt.savefig('img'+str(index))


def bootstrap_model(model, pred_model, split_data, interval=1000, confidence_q=0.1, epochs=20):
    print("bootstrapping model")
    upper_idx = int(split_data.inter_x.shape[0] / interval)
    def entropy_regularized_ce(y_true,y_pred):
        ce = losses.sparse_categorical_crossentropy(y_true, y_pred)
        entropy = -tf.reduce_sum(y_pred * tf.log(tf.clip_by_value(y_pred, 1e-10, 1-1e-10)), axis=1)
        return ce + 0.0 * entropy
    model.compile(optimizer='adam',
                  loss=[sparse_categorical_hinge(2)],
                  metrics=[metrics.sparse_categorical_accuracy])
    for i in range(upper_idx):
        cur_xs = split_data.inter_x[interval*i:interval*(i+1)]
        cur_ys = split_data.inter_y[interval*i:interval*(i+1)]
        logits = pred_model.predict(np.concatenate([cur_xs]))
        confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)
        alpha = np.quantile(confidence, confidence_q)
        indices = np.argwhere(confidence > alpha)[:, 0]
        preds = np.argmax(logits, axis=1)
        model.fit(cur_xs[indices], preds[indices], epochs=epochs, verbose=False)
        model.evaluate(cur_xs, cur_ys)
        pred_model = model
        print(np.linalg.norm(model.layers[-1].get_weights()[0]))
        # print(model.layers[-1].get_weights()[0])
        plot(cur_xs[indices], cur_ys[indices], model, index=i)
    print('Evaluate on target.')
    model.evaluate(split_data.target_val_x, split_data.target_val_y)


def main():
    # save_data()

    # dataset = faces  # When you change this, change the below 2 options.
    # interval = 1000  # Interval length for the dataset.
    # epochs = 20  # Around 200 for small datasets like Gaussian, 20 for faces. 

    # dataset = rot_mnist_0_25_15000
    # interval = 2000
    # epochs = 20
    # l2_reg = 0.02

    dataset = gauss_2D_1K_high_noise
    interval = 1000
    epochs = 200
    l2_reg = 0.5

    split_data = get_split_data(dataset)
    model = linear_model(dataset.n_classes, input_shape=dataset.input_shape, l2_reg=l2_reg)
    run_model(model, split_data, epochs=epochs)
    print(np.linalg.norm(model.layers[-1].get_weights()[0]))
    accs = rolling_accuracy(model, split_data, interval=interval, verbose=True)

    # Plot accuracies.
    # plt.plot(accs)
    # plt.xlabel('time')
    # plt.ylabel('accuracy')
    # plt.show()

    bootstrap_model(
        linear_model(dataset.n_classes, input_shape=dataset.input_shape),
        model, split_data, interval=interval, epochs=epochs)

    # bootstrap_model(linear_model(dataset.n_classes), model, split_data, interval=interval)
    # save_data()
    # Xs, Ys = make_rotated_mnist(0.0, 25.0, 30000) # load_data()
    # n_src_train = 2000
    # n_src_valid = 1000
    # n_target_test = 1000  # Do not touch the last 1000 data points.
    # n_target_val = 1000  # This is the set of target labels we use for validation.
    # n_target_unsup = 2000
    # target_end = 30000
    # n_src = n_src_train + n_src_valid
    # n_target = n_target_unsup + n_target_val + n_target_test
    # src_x, src_y = shuffle(Xs[:n_src], Ys[:n_src])
    # target_x, target_y = shuffle(
    #   Xs[target_end-n_target:target_end], Ys[target_end-n_target:target_end])
    # [src_train_x, src_val_x] = split_sizes(src_x, [n_src_train])
    # [src_train_y, src_val_y] = split_sizes(src_y, [n_src_train])
    # [target_unsup_x, target_val_x, _] = split_sizes(target_x, [n_target_unsup, n_target_val])
    # [target_unsup_y, target_val_y, _] = split_sizes(target_y, [n_target_unsup, n_target_val])
    # inter_xs, inter_ys = Xs[n_src:target_end-n_target], Ys[n_src:target_end-n_target]
    # print(np.mean(Ys))

    # # # Check all shapes.
    # # print(src_train_x.shape, src_train_y.shape)
    # # print(src_val_x.shape, src_val_y.shape)
    # # print(target_unsup_x.shape, target_val_x.shape, target_val_y.shape)


    # # src_train_x, src_train_y = Xs[:n_src_train], Ys[:n_src_train]
    # # src_valid_x, src_valid_y = (Xs[n_src_train:n_src_train+n_src_valid],
    # #                             Ys[n_src_train:n_src_train+n_src_valid])
    # # target_unsup_x = Xs[target_end-n_target_test-n_target_val-n_target_unsup:target_end-n_target_test-n_target_val]
    # # target_unsup_y = Ys[target_end-n_target_test-n_target_val-n_target_unsup:target_end-n_target_test-n_target_val]
    # # target_val_x, target_val_y = (Xs[target_end-n_target_test-n_target_val:target_end-n_target_test],
    # #                               Ys[target_end-n_target_test-n_target_val:target_end-n_target_test])
    # # inter_xs = Xs[n_src_train+n_src_valid:target_end-n_target_test-n_target_val-n_target_unsup]
    # # inter_ys = Ys[n_src_train+n_src_valid:target_end-n_target_test-n_target_val-n_target_unsup]
    # # print(inter_xs.shape)

    # def entropy_regularized_ce(y_true,y_pred):
    #   ce = losses.sparse_categorical_crossentropy(y_true, y_pred)
    #   entropy = -tf.reduce_sum(y_pred * tf.log(tf.clip_by_value(y_pred, 1e-10, 1-1e-10)), axis=1)
    #   return ce + 0.0 * entropy

    # model = linear_model(10)
    # model.compile(optimizer='adam',
    #               loss=[losses.sparse_categorical_crossentropy],
    #               metrics=[metrics.sparse_categorical_accuracy])
    # model.fit(np.concatenate([src_train_x]), np.concatenate([src_train_y]), epochs=3)
    # model.evaluate(src_val_x, src_val_y)
    # model.evaluate(target_val_x, target_val_y)

    # # Check rolling accuracy.
    # print("rolling accuracies")
    # for i in range(3, 26):
    #   cur_xs = Xs[1000*i:1000*(i+1)]
    #   cur_ys = Ys[1000*i:1000*(i+1)]
    #   model.evaluate(cur_xs, cur_ys)

    # # plt.imshow(Xs[-2])
    # # plt.show()

    # # Try bootstrapping.
    # target_model = linear_model(10)
    # pred_model = model
    # target_model.compile(optimizer='adam',
    #                    loss=entropy_regularized_ce,
    #                        metrics=[metrics.sparse_categorical_accuracy])
    # for i in range(1):
    #   logits = pred_model.predict(np.concatenate([target_unsup_x]))
    #   confidence = np.amax(logits, axis=1)
    #   alpha = np.quantile(confidence, 0.2) # 0.0  # np.median(confidence)
    #   indices = np.argwhere(confidence > alpha)[:, 0]
    #   print('num confident', len(indices))
    #   preds =  np.argmax(logits, axis=1)
    #   new_x = np.concatenate([target_unsup_x[indices]])
    #   new_y = np.concatenate([preds[indices]])
    #   target_model.fit(new_x, new_y, epochs=4, shuffle=False)
    #   target_model.evaluate(target_val_x, target_val_y)
    #   pred_model = target_model

if __name__ == "__main__":
    main()
