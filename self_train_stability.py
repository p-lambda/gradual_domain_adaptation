
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import regularizers

import logging
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

def linear_model(num_labels, input_shape, l2_reg=0.001):
    linear_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out',
    	kernel_regularizer=regularizers.l2(l2_reg))
    ])
    return linear_model


def simple_conv_model(num_labels, hidden_nodes=32, input_shape=(28,28,1), l2_reg=0.02):
    return keras.models.Sequential([
    keras.layers.Conv2D(hidden_nodes, (5,5), (2,2), activation=tf.nn.relu,
                           padding='same', input_shape=input_shape),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2,2), activation=tf.nn.relu,
                           padding='same'),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2,2), activation=tf.nn.relu,
                           padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
    # keras.layers.BatchNormalization(),
    keras.layers.Flatten(name='after_flatten'),
    # keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
    ])


def sparse_categorical_hinge(num_classes):
	def loss(y_true,y_pred):
		y_true = tf.reduce_mean(y_true, axis=1)
		y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=num_classes)
		return losses.categorical_hinge(y_true, y_pred)
	return loss

def self_train_stability():
	(train_x, train_y), (_, _) = mnist.load_data()
	input_shape = (28,28,1)
	train_x = np.expand_dims(train_x, axis=-1)
	[train_x, unsup_x, val_x], [train_y, unsup_y, val_y] = utils.split_data(
		train_x, train_y, [5000, 45000,])
	logger.info("Y shape is %s", str(train_y.shape))
	for acc in [0.6, 0.7, 0.8, 0.9]:
		model = linear_model(num_labels=10, input_shape=input_shape)
		model.compile(optimizer='adam',
		              loss=[losses.sparse_categorical_crossentropy],
		              metrics=[metrics.sparse_categorical_accuracy])
		utils.train_to_acc(model, acc, train_x, train_y, val_x, val_y)
		for i in range(10):
			utils.self_train(model, model, unsup_x, confidence_q=0.0, epochs=1)
			val_accuracy = model.evaluate(val_x, val_y, verbose=False)[1]
			logger.info("validation accuracy is %f", val_accuracy)

self_train_stability()
# Split MNIST data into labeled, unlabeled, validation
# Get MNIST models at varying accuracies
# Try varying amounts of self-train and store the losses
# Save the losses