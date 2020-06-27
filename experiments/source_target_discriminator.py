
from gradual_shift_better import compile_model
import gradual_st.utils as utils
import gradual_st.models as models
import gradual_st.datasets as datasets
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


def source_target_experiment(
    dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
    model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
    inter_as_target=None, roll_size=100):
    # Train a discriminator to classify between source and target.
    # Is this better able to identify the timestamps?
    # If inter_as_target is not None, then we use inter[inter_as_target:inter_as_target+interval]
    # as the target.
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()
    if inter_as_target is not None:
        trg_val_x = inter_x[inter_as_target:inter_as_target+interval]
    def new_model():
        model = model_func(n_classes, input_shape=input_shape)
        compile_model(model, loss)
        return model
    # Try using confidence instead.
    source_model = new_model()
    source_model.fit(src_tr_x, src_tr_y, epochs=epochs, verbose=True)
    inter_preds = source_model.predict(inter_x)[:,0]
    ranks = utils.get_confidence_ranks_by_time(inter_preds)
    rolled_ranks = utils.rolling_average(ranks, roll_size)
    plt.clf()
    plt.plot(np.arange(len(rolled_ranks)), rolled_ranks)
    plt.show()
    def new_source_target_model():
        model = model_func(2, input_shape=input_shape)
        compile_model(model, loss)
        return model
    # Make discrimination dataset.
    xs = np.concatenate([src_tr_x, trg_val_x])
    ys = np.concatenate([np.zeros(len(src_tr_x)), np.ones(len(trg_val_x))])
    source_target_model = new_source_target_model()
    source_target_model.fit(xs, ys, epochs=epochs, verbose=True)
    inter_preds = source_target_model.predict(inter_x)[:,0]
    rolled_preds = utils.rolling_average(inter_preds, roll_size)
    plt.clf()
    plt.plot(np.arange(len(rolled_preds)), rolled_preds)
    plt.show()


if __name__ == "__main__":
    # Portraits.
    # source_target_experiment()
    source_target_experiment(inter_as_target=6000)
    source_target_experiment(inter_as_target=10000)
    # MNIST.
    # source_target_experiment(
    #     dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1))
    # source_target_experiment(
    #     dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
    #     inter_as_target=20000)
