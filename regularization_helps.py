
# "Infinite" data case.
# Take n MNIST images, rotate them by angle 0
# Rotate them by successive angles, alpha more each time
# Then evaluate the accuracy as time goes by for the regularized and unregularized models (acc initially should be similar)
# Can also check if adding more steps makes things worse.

import datasets
import models
import utils
import tensorflow as tf
from tensorflow.keras import metrics
import pickle
import numpy as np


def compile_model(model, loss='ce'):
    loss = models.get_loss(loss, model.output_shape[1])
    model.compile(optimizer='adam',
                  loss=[loss],
                  metrics=[metrics.sparse_categorical_accuracy])

def rotated_mnist_regularization_experiment(
    unreg_model_func, reg_model_func, loss, save_name_base, N=2000, delta_angle=3, num_angles=21,
    retrain=False, num_runs=20):
    def run(seed):
        utils.rand_seed(seed)
        n_classes = 10
        input_shape = (28, 28, 1)
        loss = 'ce'
        epochs = 20
        interval = N
        def teacher_model():
            model = unreg_model_func(n_classes, input_shape=input_shape)
            compile_model(model, loss)
            return model
        def student_func_gen(regularize):
            iters = 0
            def student_func(teacher):
                nonlocal iters
                iters += 1
                if iters == 1 or retrain:
                    if regularize:
                        model = reg_model_func(n_classes, input_shape=input_shape)
                    else:
                        model = unreg_model_func(n_classes, input_shape=input_shape)
                    compile_model(model, loss)
                    return model
                return teacher
            return student_func
        # Get data.
        (train_x, train_y), _ = datasets.get_preprocessed_mnist()
        orig_x, orig_y = train_x[:N], train_y[:N]
        inter_x, inter_y = datasets.make_population_rotated_dataset(
            orig_x, orig_y, delta_angle, num_angles)
        trg_x, trg_y = inter_x[-N:], inter_y[-N:]
        # Train source model.
        source_model = teacher_model()
        source_model.fit(orig_x, orig_y, epochs=epochs, verbose=False)
        _, src_acc = source_model.evaluate(orig_x, orig_y)
        _, target_acc = source_model.evaluate(trg_x, trg_y)
        def gradual_train(regularize):
            teacher = teacher_model()
            teacher.set_weights(source_model.get_weights())
            accuracies, student = utils.gradual_self_train(
                student_func_gen(regularize=regularize), teacher, inter_x, inter_y, interval, epochs=epochs)
            _, acc = student.evaluate(trg_x, trg_y)
            accuracies.append(acc)
            return accuracies
        # Regularized gradual self-training.
        print("\n\n Regularized gradual self-training:")
        reg_accuracies = gradual_train(regularize=True)
        # Unregularized
        print("\n\n Unregularized gradual self-training:")
        unreg_accuracies = gradual_train(regularize=False)
        return src_acc, target_acc, reg_accuracies, unreg_accuracies
    results = []
    for i in range(num_runs):
        results.append(run(i))
    save_name = (save_name_base + '_' + str(N) + '_' + str(delta_angle) + '_' + str(num_angles) +
                 '.dat')
    print('Saving to ' + save_name)
    pickle.dump(results, open(save_name, "wb"))


def rotated_mnist_regularization_results(save_name):
    results = pickle.load(open(save_name, "rb"))
    src_accs, target_accs, reg_accs, unreg_accs = [], [], [], []
    for src_acc, target_acc, reg_accuracies, unreg_accuracies in results:
        src_accs.append(100 * src_acc)
        target_accs.append(100 * target_acc)
        reg_accs.append(100 * reg_accuracies[-1])
        unreg_accs.append(100 * unreg_accuracies[-1])
    num_runs = len(src_accs)
    mult = 1.645  # For 90% confidence intervals
    print("Source accuracy (%): ", np.mean(src_accs),
          mult * np.std(src_accs) / np.sqrt(num_runs))
    print("Target accuracy (%): ", np.mean(target_accs),
          mult * np.std(target_accs) / np.sqrt(num_runs))
    print("Reg accuracy (%): ", np.mean(reg_accs),
          mult * np.std(reg_accs) / np.sqrt(num_runs))
    print("Unreg accuracy (%): ", np.mean(unreg_accs),
          mult * np.std(unreg_accs) / np.sqrt(num_runs))

if __name__ == "__main__":
    rotated_mnist_regularization_experiment(
        models.unregularized_softmax_conv_model, models.simple_softmax_conv_model, 'ce',
        save_name_base='saved_files/inf_reg_mnist', N=2000, delta_angle=3, num_angles=20,
        retrain=False, num_runs=5)
    rotated_mnist_regularization_results('saved_files/inf_reg_mnist_2000_3_20.dat')
