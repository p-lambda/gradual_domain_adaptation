
import utils
import models
import datasets
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
import pickle


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


def run_experiment(
    dataset_func, n_classes, input_shape, save_file, model_func=models.simple_softmax_conv_model,
    interval=2000, epochs=10, loss='ce', num_runs=20):
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()
    num_repeats = int(inter_x.shape[0] / interval)
    def new_model():
        model = model_func(n_classes, input_shape=input_shape)
        compile_model(model, loss)
        return model
    def student_func(teacher):
        return teacher
    def run(seed):
        utils.rand_seed(seed)
        trg_eval_x = trg_val_x
        trg_eval_y = trg_val_y
        # Train source model.
        source_model = new_model()
        source_model.fit(src_tr_x, src_tr_y, epochs=epochs, verbose=False)
        _, src_acc = source_model.evaluate(src_val_x, src_val_y)
        _, target_acc = source_model.evaluate(trg_eval_x, trg_eval_y)
        # Gradual self-training.
        print("\n\n Gradual self-training:")
        teacher = new_model()
        teacher.set_weights(source_model.get_weights())
        gradual_accuracies, student = utils.gradual_self_train(
            student_func, teacher, inter_x, inter_y, interval, epochs=epochs)
        _, acc = student.evaluate(trg_eval_x, trg_eval_y)
        gradual_accuracies.append(acc)
        # Train to target.
        print("\n\n Direct boostrap to target:")
        teacher = new_model()
        teacher.set_weights(source_model.get_weights())
        target_accuracies, _ = utils.self_train(
            student_func, teacher, dir_inter_x, epochs=epochs, target_x=trg_eval_x,
            target_y=trg_eval_y, repeats=num_repeats)
        print("\n\n Direct boostrap to all unsup data:")
        teacher = new_model()
        teacher.set_weights(source_model.get_weights())
        all_accuracies, _ = utils.self_train(
            student_func, teacher, inter_x, epochs=epochs, target_x=trg_eval_x,
            target_y=trg_eval_y, repeats=num_repeats)
        return src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies
    results = []
    for i in range(num_runs):
        results.append(run(i))
    print('Saving to ' + save_file)
    pickle.dump(results, open(save_file, "wb"))


def experiment_results(save_name):
    results = pickle.load(open(save_name, "rb"))
    src_accs, target_accs = [], []
    final_graduals, final_targets, final_alls = [], [], []
    best_targets, best_alls = [], []
    for src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies in results:
        src_accs.append(100 * src_acc)
        target_accs.append(100 * target_acc)
        final_graduals.append(100 * gradual_accuracies[-1])
        final_targets.append(100 * target_accuracies[-1])
        final_alls.append(100 * all_accuracies[-1])
        best_targets.append(100 * np.max(target_accuracies))
        best_alls.append(100 * np.max(all_accuracies))
    num_runs = len(src_accs)
    mult = 1.645  # For 90% confidence intervals
    print("Source accuracy (%): ", np.mean(src_accs),
          mult * np.std(src_accs) / np.sqrt(num_runs))
    print("Target accuracy (%): ", np.mean(target_accs),
          mult * np.std(target_accs) / np.sqrt(num_runs))
    print("Gradual accuracy (%): ", np.mean(final_graduals),
          mult * np.std(final_graduals) / np.sqrt(num_runs))
    print("Target accuracy (%): ", np.mean(final_targets),
          mult * np.std(final_targets) / np.sqrt(num_runs))
    print("All accuracy (%): ", np.mean(final_alls),
          mult * np.std(final_alls) / np.sqrt(num_runs))
    print("Best of Target accuracy (%): ", np.mean(best_targets),
          mult * np.std(best_targets) / np.sqrt(num_runs))
    print("Best of All accuracy (%): ", np.mean(best_alls),
          mult * np.std(best_alls) / np.sqrt(num_runs))


def rotated_mnist_60_conv_experiment():
    def data_func():
        (train_x, train_y), (test_x, test_y) = datasets.get_preprocessed_mnist()
        return datasets.make_rotated_dataset(
            train_x, train_y, test_x, test_y, [0.0, 5.0], [5.0, 60.0], [55.0, 60.0],
            5000, 6000, 48000, 50000)
    run_experiment(
        dataset_func=data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        num_runs=5)


def portraits_conv_experiment():
    def data_func():
        return datasets.make_portraits_data(1000, 1000, 14000, 2000, 1000, 1000)
    run_experiment(
        dataset_func=data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        num_runs=5)


def gaussian_linear_experiment():
    d = 100
    def data_func():
        return datasets.make_high_d_gaussian_data(
            d=d, min_var=0.05, max_var=0.1,
            source_alphas=[0.0, 0.0], inter_alphas=[0.0, 1.0], target_alphas=[1.0, 1.0],
            n_src_tr=500, n_src_val=1000, n_inter=5000, n_trg_val=1000, n_trg_tst=1000)
    run_experiment(
        dataset_func=data_func, n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian.dat',
        model_func=models.linear_softmax_model, interval=500, epochs=100, loss='ce',
        num_runs=5)


# def rotated_mnist_60_conv_experiment(seed=1):
#     rand_seed(seed)
#     dataset = rot_mnist_0_60_50000
#     model_func = models.simple_softmax_conv_model
#     interval = 2000
#     epochs = 20
#     loss = 'ce'
#     split_data = datasets.get_split_data(rot_mnist_0_60_50000)
#     def new_model():
#         model = model_func(dataset.n_classes, input_shape=dataset.input_shape)
#         compile_model(model, loss)
#         return model
#     def student_func(teacher):
#         return teacher
#     # Train source model.
#     source_model = new_model()
#     src_acc, target_acc = train_model_source(source_model, split_data, epochs)
#     # Gradual self-training.
#     print("\n\n Gradual bootstrap:")
#     teacher = new_model()
#     teacher.set_weights(source_model.get_weights())
#     unsup_x = np.concatenate([split_data.inter_x, split_data.target_unsup_x])
#     debug_y = np.concatenate([split_data.inter_y, split_data.debug_target_unsup_y])
#     gradual_accuracies, student = utils.gradual_self_train(
#         student_func, teacher, unsup_x, debug_y, interval, epochs=epochs)
#     _, acc = student.evaluate(split_data.target_val_x, split_data.target_val_y)
#     gradual_accuracies.append(acc)
#     # Train to target.
#     print("\n\n Direct boostrap to target:")
#     teacher = new_model()
#     teacher.set_weights(source_model.get_weights())
#     # TODO: get more rotated data at target angles.
#     # Get target angles.
#     # Get mnist dataset
#     # Rotate intermediate to have those angles (iid?)
#     # TODO: Do self-training on target angles
#     # Call self-train on all these new targets
#     # Train to all unsupervised data.
#     print("\n\n Direct boostrap to all unsup data:")
#     teacher = new_model()
#     teacher.set_weights(source_model.get_weights())
#     all_accuracies = utils.self_train(
#         student_func, teacher, unsup_x, epochs=epochs, target_x=split_data.target_val_x,
#         target_y=split_data.target_val_y)


if __name__ == "__main__":
    # portraits_conv_experiment()
    # experiment_results('saved_files/portraits.dat')
    # rotated_mnist_60_conv_experiment()
    # experiment_results('saved_files/rot_mnist_60_conv.dat')
    gaussian_linear_experiment()
    # experiment_results('saved_files/gaussian.dat')
