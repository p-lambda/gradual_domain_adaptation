
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist


def simple_conv_model(num_labels, hidden_nodes=64, input_shape=(28,28,1), l2_reg=0.0):
    return keras.models.Sequential([
    keras.layers.Conv2D(hidden_nodes, (5,5), (2,2), activation=tf.nn.relu,
                           padding='same', input_shape=input_shape),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2,2), activation=tf.nn.relu,
                           padding='same'),
    # keras.layers.SpatialDropout2D(0.5),
    # keras.layers.Conv2D(hidden_nodes, (5,5), (2,2), activation=tf.nn.relu,
    #                        padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(name='after_flatten'),
    # keras.layers.Dense(150, activation=tf.nn.relu),
    # keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
    ])


def split_dataset(num_labeled, num_classes, train_x, train_y):
    assert(num_labeled % num_classes == 0)
    assert(np.min(train_y) == 0 and np.max(train_y) == num_classes - 1)
    per_class = num_labeled // 9
    labeled_x, labeled_y = [], []
    unlabeled_x, unlabeled_y = [], []
    for i in range(num_classes):
        class_filter = (train_y == i)
        class_x = train_x[class_filter]
        class_y = train_y[class_filter]
        labeled_x.append(class_x[:per_class])
        labeled_y.append(class_y[:per_class])
        unlabeled_x.append(class_x[per_class:])
        unlabeled_y.append(class_y[per_class:])
    labeled_x = np.concatenate(labeled_x)
    labeled_y = np.concatenate(labeled_y)
    unlabeled_x = np.concatenate(unlabeled_x)
    unlabeled_y = np.concatenate(unlabeled_y)
    assert(labeled_x.shape[0] == labeled_y.shape[0])
    assert(labeled_x.shape[1:] == train_x.shape[1:])
    return labeled_x, labeled_y, unlabeled_x, unlabeled_y



def pseudolabel(model, num_labeled, train_x, train_y, test_x, test_y):
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # labeled_x, labeled_y = train_x[:num_labeled], train_y[:num_labeled]
    labeled_x, labeled_y, unlabeled_x, _ = split_dataset(
        num_labeled, 10, train_x, train_y)
    train_x = train_x / 255.0
    print(labeled_x.shape, labeled_y.shape)
    print(labeled_y[:10], labeled_y[100:110])
    print([np.sum(labeled_y == i) for i in range(10)])
    model.fit(labeled_x, labeled_y, epochs=10)
    model.evaluate(test_x, test_y)

    # Unlabeled.
    confidence_q = 0.1
    epochs = 10
    for i in range(epochs):
        logits = model.predict(np.concatenate([unlabeled_x]))
        confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)
        alpha = np.quantile(confidence, confidence_q)
        indices = np.argwhere(confidence >= alpha)[:, 0]
        preds = np.argmax(logits, axis=1)
        model.fit(unlabeled_x[indices], preds[indices], epochs=1, verbose=False)
        model.evaluate(test_x, test_y)


def main():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = np.expand_dims(train_x, axis=-1)
    test_x = np.expand_dims(test_x, axis=-1)
    conv_model = simple_conv_model(10)
    pseudolabel(conv_model, 1000, train_x, train_y, test_x, test_y)

main()
