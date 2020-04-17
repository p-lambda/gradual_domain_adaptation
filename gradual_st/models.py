
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras import losses


# Models.

def linear_model(num_labels, input_shape, l2_reg=0.02):
    linear_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(num_labels, activation=None, name='out',
        kernel_regularizer=regularizers.l2(l2_reg))
    ])
    return linear_model


def linear_softmax_model(num_labels, input_shape, l2_reg=0.02):
    linear_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out',
        kernel_regularizer=regularizers.l2(l2_reg))
    ])
    return linear_model


def mlp_softmax_model(num_labels, input_shape, l2_reg=0.0, dropout=0.5):
    linear_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(32, activation=tf.nn.relu,
        kernel_regularizer=regularizers.l2(l2_reg)),
    keras.layers.Dense(32, activation=tf.nn.relu,
        kernel_regularizer=regularizers.l2(l2_reg)),
    keras.layers.Dropout(dropout),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out',
        kernel_regularizer=regularizers.l2(l2_reg))
    ])
    return linear_model


def simple_softmax_conv_model(num_labels, hidden_nodes=32, input_shape=(28,28,1), l2_reg=0.0, dropout=0.5):
    return keras.models.Sequential([
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='same', input_shape=input_shape,
                           kernel_regularizer=regularizers.l2(l2_reg)),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='same',
                           kernel_regularizer=regularizers.l2(l2_reg)),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='same',
                           kernel_regularizer=regularizers.l2(l2_reg)),
    keras.layers.Dropout(dropout),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(name='after_flatten'),
    # keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out',
        kernel_regularizer=regularizers.l2(l2_reg))
    ])


def deeper_softmax_conv_model(num_labels, hidden_nodes=32, input_shape=(28,28,1), l2_reg=0.0):
    return keras.models.Sequential([
    keras.layers.Conv2D(hidden_nodes, (5,5), (1, 1), activation=tf.nn.relu,
                           padding='same', input_shape=input_shape),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='same', input_shape=input_shape),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='same'),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='same'),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(name='after_flatten'),
    # keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
    ])


def unregularized_softmax_conv_model(num_labels, hidden_nodes=32, input_shape=(28,28,1), l2_reg=0.0):
    return keras.models.Sequential([
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='same', input_shape=input_shape),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='same'),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='same'),
    keras.layers.Flatten(name='after_flatten'),
    # keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
    ])


def keras_mnist_model(num_labels, input_shape=(28,28,1)):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_labels, activation='softmax'))
    return model


def unregularized_keras_mnist_model(num_labels, input_shape=(28,28,1)):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(num_labels, activation='softmax'))
    return model


def papernot_softmax_model(num_labels, input_shape=(28,28,1), l2_reg=0.0):
    papernot_conv_model = keras.models.Sequential([
     keras.layers.Conv2D(64, (8, 8), (2,2), activation=tf.nn.relu,
                            padding='same', input_shape=input_shape),
     keras.layers.Conv2D(128, (6,6), (2,2), activation=tf.nn.relu,
                            padding='valid'),
     keras.layers.Conv2D(128, (5,5), (1,1), activation=tf.nn.relu,
                            padding='valid'),
     keras.layers.BatchNormalization(),
     keras.layers.Flatten(name='after_flatten'),
     keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
    ])
    return papernot_conv_model


# Losses.

def sparse_categorical_hinge(num_classes):
    def loss(y_true,y_pred):
        y_true = tf.reduce_mean(y_true, axis=1)
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=num_classes)
        return losses.categorical_hinge(y_true, y_pred)
    return loss


def sparse_categorical_ramp(num_classes):
    def loss(y_true,y_pred):
        y_true = tf.reduce_mean(y_true, axis=1)
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=num_classes)
        return tf.sqrt(losses.categorical_hinge(y_true, y_pred))
    return loss


def get_loss(loss_name, num_classes):
    if loss_name == 'hinge':
        loss = sparse_categorical_hinge(num_classes)
    elif loss_name == 'ramp':
        loss = sparse_categorical_ramp(num_classes)
    elif loss_name == 'ce':
        loss = losses.sparse_categorical_crossentropy
    elif loss_name == 'categorical_ce':
        loss = losses.categorical_crossentropy
    else:
        raise ValueError("Cannot parse loss %s", loss_name)
    return loss
