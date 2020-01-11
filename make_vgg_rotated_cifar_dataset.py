
import argparse
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from scipy import ndimage
import pickle

import cifar10vgg

def linear_model(num_labels, input_shape, l2_reg=0.02):
    linear_model = Sequential([
    Flatten(input_shape=input_shape),
    Dense(num_labels, activation='softmax', name='out'),
    ])
    return linear_model

parser = argparse.ArgumentParser()
parser.add_argument('--save_file', default='rotated_cifar_reps.dat', type=str,
                    help='Name of file to save logits, labels pair.')


def make_rotated_dataset(dataset, start_angle, end_angle, num_points):
    images, labels = [], []
    (train_x, train_y), (_, _) = dataset.load_data()
    # train_x = train_x / 255.0
    assert(num_points < train_x.shape[0])
    indices = np.random.choice(train_x.shape[0], size=num_points, replace=False)
    for i in range(num_points):
        angle = float(end_angle - start_angle) / num_points * i + start_angle
        idx = indices[i]
        img = ndimage.rotate(train_x[idx], angle, reshape=False)
        images.append(img)
        labels.append(train_y[idx])
    return np.array(images), np.array(labels)


def make_rotated_cifar10(start_angle, end_angle, num_points):
    return make_rotated_dataset(cifar10, start_angle, end_angle, num_points)   


def make_dataset(filename='rotated_cifar_features.dat'):
    args = parser.parse_args()
    xs, ys = make_rotated_cifar10(0.0, 20.0, 29000)
    vggclass = cifar10vgg.cifar10vgg()
    model = vggclass.get_model()
    features = model.layers[-22].output
    feature_model = Model(inputs=model.inputs, outputs=features)
    normalized_xs = vggclass.normalize_production(xs)
    features = feature_model.predict(normalized_xs)
    pickle.dump((features, ys), open(filename, "wb"))
    print(features.shape) 

def test_dataset(filename='rotated_cifar_features.dat'):
    features, labels = pickle.load(open(filename, "rb"))
    model = linear_model(10, features.shape[1:])
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
    model.fit(features, labels, epochs=10)


if __name__ == "__main__":
    make_dataset('rotated_cifar_conv_features.dat')
    # test_dataset()
