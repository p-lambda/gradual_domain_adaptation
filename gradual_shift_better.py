
import utils
import models
import datasets
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
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


def run_experiment(
    dataset_func, n_classes, input_shape, save_file, model_func=models.simple_softmax_conv_model,
    interval=2000, epochs=10, loss='ce', soft=False, conf_q=0.1, num_runs=20, num_repeats=None):
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()
    if soft:
        src_tr_y = to_categorical(src_tr_y)
        src_val_y = to_categorical(src_val_y)
        trg_eval_y = to_categorical(trg_eval_y)
        dir_inter_y = to_categorical(dir_inter_y)
        inter_y = to_categorical(inter_y)
        trg_test_y = to_categorical(trg_test_y)
    if num_repeats is None:
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
            student_func, teacher, inter_x, inter_y, interval, epochs=epochs, soft=soft,
            confidence_q=conf_q)
        _, acc = student.evaluate(trg_eval_x, trg_eval_y)
        gradual_accuracies.append(acc)
        # Train to target.
        print("\n\n Direct boostrap to target:")
        teacher = new_model()
        teacher.set_weights(source_model.get_weights())
        target_accuracies, _ = utils.self_train(
            student_func, teacher, dir_inter_x, epochs=epochs, target_x=trg_eval_x,
            target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
        print("\n\n Direct boostrap to all unsup data:")
        teacher = new_model()
        teacher.set_weights(source_model.get_weights())
        all_accuracies, _ = utils.self_train(
            student_func, teacher, inter_x, epochs=epochs, target_x=trg_eval_x,
            target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
        return src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies
    results = []
    for i in range(num_runs):
        results.append(run(i))
    print('Saving to ' + save_file)
    pickle.dump(results, open(save_file, "wb"))


def learn_gradual_structure_experiment(
    dataset_func, n_classes, input_shape, save_file, model_func=models.simple_softmax_conv_model,
    interval=2000, epochs=10, loss='ce', soft=False, conf_q=0.1, num_runs=20, num_repeats=None):
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()
    if soft:
        src_tr_y = to_categorical(src_tr_y)
        src_val_y = to_categorical(src_val_y)
        trg_eval_y = to_categorical(trg_eval_y)
        dir_inter_y = to_categorical(dir_inter_y)
        inter_y = to_categorical(inter_y)
        trg_test_y = to_categorical(trg_test_y)
    if num_repeats is None:
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
        # Learn gradual structure
        print("\n\n Learn gradual structure")
        teacher = new_model()
        teacher.set_weights(source_model.get_weights())
        learn_accuracies, student = utils.self_train_learn_gradual(
            student_func, teacher, inter_x, inter_y, num_new_pts=interval, epochs=epochs, soft=soft)
        _, acc = student.evaluate(trg_eval_x, trg_eval_y)
        learn_accuracies.append(acc)
        # # Gradual self-training.
        # print("\n\n Gradual self-training:")
        # teacher = new_model()
        # teacher.set_weights(source_model.get_weights())
        # gradual_accuracies, student = utils.gradual_self_train(
        #     student_func, teacher, inter_x, inter_y, interval, epochs=epochs, soft=soft,
        #     confidence_q=conf_q)
        # _, acc = student.evaluate(trg_eval_x, trg_eval_y)
        # gradual_accuracies.append(acc)
        
        # print("\n\n Direct boostrap to all unsup data:")
        # teacher = new_model()
        # teacher.set_weights(source_model.get_weights())
        # all_accuracies, _ = utils.self_train(
        #     student_func, teacher, inter_x, epochs=epochs, target_x=trg_eval_x,
        #     target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
        # return src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies
        return learn_accuracies
    results = []
    for i in range(num_runs):
        results.append(run(i))
        print(results[-1])
    pring(results)
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
    print("\nNon-adaptive accuracy on source (%): ", np.mean(src_accs),
          mult * np.std(src_accs) / np.sqrt(num_runs))
    print("Non-adaptive accuracy on target (%): ", np.mean(target_accs),
          mult * np.std(target_accs) / np.sqrt(num_runs))
    print("Gradual self-train accuracy (%): ", np.mean(final_graduals),
          mult * np.std(final_graduals) / np.sqrt(num_runs))
    print("Target self-train accuracy (%): ", np.mean(final_targets),
          mult * np.std(final_targets) / np.sqrt(num_runs))
    print("All self-train accuracy (%): ", np.mean(final_alls),
          mult * np.std(final_alls) / np.sqrt(num_runs))
    print("Best of Target self-train accuracies (%): ", np.mean(best_targets),
          mult * np.std(best_targets) / np.sqrt(num_runs))
    print("Best of All self-train accuracies (%): ", np.mean(best_alls),
          mult * np.std(best_alls) / np.sqrt(num_runs))


def rotated_mnist_60_conv_learn_structure_experiment():
    learn_gradual_structure_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv_learn_structure.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def rotated_mnist_60_conv_experiment():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def portraits_conv_experiment():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def gaussian_linear_experiment():
    d = 100        
    run_experiment(
        dataset_func=lambda: datasets.gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian.dat',
        model_func=models.linear_softmax_model, interval=500, epochs=100, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


# Ablations below.

def rotated_mnist_60_conv_experiment_noconf():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv_noconf.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        soft=False, conf_q=0.0, num_runs=5)


def portraits_conv_experiment_noconf():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_noconf.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.0, num_runs=5)


def gaussian_linear_experiment_noconf():
    d = 100        
    run_experiment(
        dataset_func=lambda: datasets.gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian_noconf.dat',
        model_func=models.linear_softmax_model, interval=500, epochs=100, loss='ce',
        soft=False, conf_q=0.0, num_runs=5)


def portraits_64_conv_experiment():
    run_experiment(
        dataset_func=datasets.portraits_64_data_func, n_classes=2, input_shape=(64, 64, 1),
        save_file='saved_files/portraits_64.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def dialing_ratios_mnist_experiment():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_dialing_ratios_data_func,
        n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/dialing_rot_mnist_60_conv.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def portraits_conv_experiment_more():
    run_experiment(
        dataset_func=datasets.portraits_data_func_more, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_more.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def rotated_mnist_60_conv_experiment_smaller_interval():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv_smaller_interval.dat',
        model_func=models.simple_softmax_conv_model, interval=1000, epochs=10, loss='ce',
        soft=False, conf_q=0.1, num_runs=5, num_repeats=7)


def portraits_conv_experiment_smaller_interval():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_smaller_interval.dat',
        model_func=models.simple_softmax_conv_model, interval=1000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5, num_repeats=7)


def gaussian_linear_experiment_smaller_interval():
    d = 100        
    run_experiment(
        dataset_func=lambda: datasets.gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian_smaller_interval.dat',
        model_func=models.linear_softmax_model, interval=250, epochs=100, loss='ce',
        soft=False, conf_q=0.1, num_runs=5, num_repeats=7)



def rotated_mnist_60_conv_experiment_more_epochs():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv_more_epochs.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=15, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def portraits_conv_experiment_more_epochs():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_more_epochs.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=30, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def gaussian_linear_experiment_more_epochs():
    d = 100        
    run_experiment(
        dataset_func=lambda: datasets.gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian_more_epochs.dat',
        model_func=models.linear_softmax_model, interval=500, epochs=150, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


if __name__ == "__main__":
    # Learn gradual structure.
    rotated_mnist_60_conv_learn_structure_experiment()
    experiment_results('saved_files/rot_mnist_60_conv_learn_structure.dat')

    # # Main paper experiments.
    # portraits_conv_experiment()
    # print("Portraits conv experiment")
    # experiment_results('saved_files/portraits.dat')
    # rotated_mnist_60_conv_experiment()
    # print("Rot MNIST conv experiment")
    # experiment_results('saved_files/rot_mnist_60_conv.dat')
    # gaussian_linear_experiment()
    # print("Gaussian linear experiment")
    # experiment_results('saved_files/gaussian.dat')
    # print("Dialing MNIST ratios conv experiment")
    # dialing_ratios_mnist_experiment()
    # experiment_results('saved_files/dialing_rot_mnist_60_conv.dat')

    # # Without confidence thresholding.
    # portraits_conv_experiment_noconf()
    # print("Portraits conv experiment no confidence thresholding")
    # experiment_results('saved_files/portraits_noconf.dat')
    # rotated_mnist_60_conv_experiment_noconf()
    # print("Rot MNIST conv experiment no confidence thresholding")
    # experiment_results('saved_files/rot_mnist_60_conv_noconf.dat')
    # gaussian_linear_experiment_noconf()
    # print("Gaussian linear experiment no confidence thresholding")
    # experiment_results('saved_files/gaussian_noconf.dat')

    # # Try predicting for next set of data points on portraits.
    # portraits_conv_experiment_more()
    # print("Portraits next datapoints conv experiment")
    # experiment_results('saved_files/portraits_more.dat')

    # # Try smaller window sizes.
    # portraits_conv_experiment_smaller_interval()
    # print("Portraits conv experiment smaller window")
    # experiment_results('saved_files/portraits_smaller_interval.dat')
    # rotated_mnist_60_conv_experiment_smaller_interval()
    # print("Rot MNIST conv experiment smaller window")
    # experiment_results('saved_files/rot_mnist_60_conv_smaller_interval.dat')
    # gaussian_linear_experiment_smaller_interval()
    # print("Gaussian linear experiment smaller window")
    # experiment_results('saved_files/gaussian_smaller_interval.dat')

    # # Try training more epochs.
    # portraits_conv_experiment_more_epochs()
    # print("Portraits conv experiment train longer")
    # experiment_results('saved_files/portraits_more_epochs.dat')
    # rotated_mnist_60_conv_experiment_more_epochs()
    # print("Rot MNIST conv experiment train longer")
    # experiment_results('saved_files/rot_mnist_60_conv_more_epochs.dat')
    # gaussian_linear_experiment_more_epochs()
    # print("Gaussian linear experiment train longer")
    # experiment_results('saved_files/gaussian_more_epochs.dat')
