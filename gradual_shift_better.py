
import utils
import models
import datasets
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist


def rand_seed(seed):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


rot_mnist_0_60_50000 = datasets.Dataset(
    get_data = lambda: datasets.make_rotated_mnist(0.0, 60.0, 50000),
    n_src_train = 3000,
    n_src_valid = 1000,
    n_target_unsup = 2000,
    n_target_val = 1000,
    n_target_test = 1000,
    target_end=50000,
    n_classes=10,
    input_shape=(28,28,1),
)


def compile_model(model, loss='ce'):
    loss = models.get_loss(loss, model.output_shape[1])
    model.compile(optimizer='adam',
                  loss=[loss],
                  metrics=[metrics.sparse_categorical_accuracy])


def train_model_source(model, split_data, epochs=1000):
    model.fit(split_data.src_train_x, split_data.src_train_y, epochs=epochs, verbose=False)
    print("Source accuracy:")
    _, src_acc = model.evaluate(split_data.src_val_x, split_data.src_val_y)
    print("Target accuracy:")
    _, target_acc = model.evaluate(split_data.target_val_x, split_data.target_val_y)
    return src_acc, target_acc



# Source distribution: angles a to b
# Target distribution: angle c to d
# Source_train, source_valid
# target_train, target_valid
# target_test (test set, from angle c to d)
# These are all held constant.
# Intermediate examples, in direct adaptation we get more samples from target dist
# For gradual shift, adapt to all, get examples from b to d
# For direct adaptation, get examples sampled from c to d (target dist) - in reality not available
# Can also compare to using intermediate examples from c to do.


def new_rotated_mnist_60_conv_experiment(seed=1):
    (train_x, train_y), (test_x, test_y) = datasets.get_preprocessed_mnist()
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = datasets.make_rotated_dataset(
        train_x, train_y, test_x, test_y, [0.0, 5.0], [5.0, 50.0], [45.0, 50.0],
        5000, 6000, 48000, 50000)
    model_func = models.simple_softmax_conv_model
    interval = 2000
    epochs = 10
    loss = 'ce'
    n_classes = 10
    input_shape = (28,28,1)
    def new_model():
        model = model_func(n_classes, input_shape=input_shape)
        compile_model(model, loss)
        return model
    def student_func(teacher):
        return teacher
    def run(seed):
        rand_seed(seed)
        trg_eval_x = trg_val_x
        trg_eval_y = trg_val_y
        # Train source model.
        source_model = new_model()
        source_model.fit(src_tr_x, src_tr_y, epochs=epochs, verbose=False)
        _, src_acc = source_model.evaluate(src_val_x, src_val_y)
        _, target_acc = source_model.evaluate(trg_eval_x, trg_eval_y)
        # Gradual self-training.
        # print("\n\n Gradual self-training:")
        # teacher = new_model()
        # teacher.set_weights(source_model.get_weights())
        # gradual_accuracies, student = utils.gradual_self_train(
        #     student_func, teacher, inter_x, inter_y, interval, epochs=epochs)
        # _, acc = student.evaluate(trg_eval_x, trg_eval_y)
        # gradual_accuracies.append(acc)
        # Train to target.
        print("\n\n Direct boostrap to target:")
        teacher = new_model()
        teacher.set_weights(source_model.get_weights())
        utils.self_train(
            student_func, teacher, dir_inter_x, epochs=epochs, target_x=trg_eval_x,
            target_y=trg_eval_y, repeats=20)
        print("\n\n Direct boostrap to all unsup data:")
        teacher = new_model()
        teacher.set_weights(source_model.get_weights())
        all_accuracies = utils.self_train(
            student_func, teacher, unsup_x, epochs=epochs, target_x=split_data.target_val_x,
            target_y=split_data.target_val_y)
    run(1)


def rotated_mnist_60_conv_experiment(seed=1):
    rand_seed(seed)
    dataset = rot_mnist_0_60_50000
    model_func = models.simple_softmax_conv_model
    interval = 2000
    epochs = 20
    loss = 'ce'
    split_data = datasets.get_split_data(rot_mnist_0_60_50000)
    def new_model():
        model = model_func(dataset.n_classes, input_shape=dataset.input_shape)
        compile_model(model, loss)
        return model
    def student_func(teacher):
        return teacher
    # Train source model.
    source_model = new_model()
    src_acc, target_acc = train_model_source(source_model, split_data, epochs)
    # Gradual self-training.
    print("\n\n Gradual bootstrap:")
    teacher = new_model()
    teacher.set_weights(source_model.get_weights())
    unsup_x = np.concatenate([split_data.inter_x, split_data.target_unsup_x])
    debug_y = np.concatenate([split_data.inter_y, split_data.debug_target_unsup_y])
    gradual_accuracies, student = utils.gradual_self_train(
        student_func, teacher, unsup_x, debug_y, interval, epochs=epochs)
    _, acc = student.evaluate(split_data.target_val_x, split_data.target_val_y)
    gradual_accuracies.append(acc)
    # Train to target.
    print("\n\n Direct boostrap to target:")
    teacher = new_model()
    teacher.set_weights(source_model.get_weights())
    # TODO: get more rotated data at target angles.
    # Get target angles.
    # Get mnist dataset
    # Rotate intermediate to have those angles (iid?)
    # TODO: Do self-training on target angles
    # Call self-train on all these new targets
    # Train to all unsupervised data.
    print("\n\n Direct boostrap to all unsup data:")
    teacher = new_model()
    teacher.set_weights(source_model.get_weights())
    all_accuracies = utils.self_train(
        student_func, teacher, unsup_x, epochs=epochs, target_x=split_data.target_val_x,
        target_y=split_data.target_val_y)


if __name__ == "__main__":
    new_rotated_mnist_60_conv_experiment()
    # rotated_mnist_60_conv_experiment()
