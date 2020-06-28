
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def compile_model(model, loss='ce'):
    loss = models.get_loss(loss, model.output_shape[1])
    model.compile(optimizer='adam',
                  loss=[loss],
                  metrics=[metrics.sparse_categorical_accuracy])


def oracle_performance(dataset_func, n_classes, input_shape, save_file, model_func=models.simple_softmax_conv_model,
    epochs=10, loss='ce'):
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()
    def new_model():
        model = model_func(n_classes, input_shape=input_shape)
        compile_model(model, loss)
        return model
    def run(seed):
        utils.rand_seed(seed)
        trg_eval_x = trg_val_x
        trg_eval_y = trg_val_y
        # Train source model.
        source_model = new_model()
        xs = np.concatenate([dir_inter_x])
        ys = np.concatenate([dir_inter_y])
        source_model.fit(xs, ys, epochs=epochs, verbose=True)
        _, target_acc = source_model.evaluate(trg_eval_x, trg_eval_y)
        print(target_acc)
        return src_acc, target_acc
    return run(0)


def make_training_data(xs, ys):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen.fit(xs)
    return datagen.flow(xs, ys)


def make_test_data(xs, ys):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)
    datagen.fit(xs)
    return datagen.flow(xs, ys)


def run_source_experiment(
    dataset_func, n_classes, input_shape, save_file, model_func=models.simple_softmax_conv_model,
    epochs=10, loss='ce', num_runs=20, augmentation=True):
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()
    def new_model():
        model = model_func(n_classes, input_shape=input_shape)
        compile_model(model, loss)
        return model
    def run(seed):
        utils.rand_seed(seed)
        trg_eval_x = trg_val_x
        trg_eval_y = trg_val_y
        # Train source model.
        source_model = new_model()
        if augmentation:
            src_tr_data = make_training_data(src_tr_x, src_tr_y)
            src_val_data = make_test_data(src_val_x, src_val_y)
            trg_eval_data = make_test_data(trg_eval_x, trg_eval_y)
            source_model.fit(src_tr_data, epochs=epochs, verbose=False)
            _, src_acc = source_model.evaluate(src_val_data)
            _, target_acc = source_model.evaluate(trg_eval_data)
        else:
            source_model.fit(src_tr_x, src_tr_y, epochs=epochs, verbose=False)
            _, src_acc = source_model.evaluate(src_val_x, src_val_y)
            _, target_acc = source_model.evaluate(trg_eval_x, trg_eval_y)
        return src_acc, target_acc
    results = []
    for i in range(num_runs):
        results.append(run(i))
    print('Saving to ' + save_file)
    pickle.dump(results, open(save_file, "wb"))


def source_experiment_results(save_name):
    results = pickle.load(open(save_name, "rb"))
    src_accs, target_accs = [], []
    for src_acc, target_acc in results:
        src_accs.append(100 * src_acc)
        target_accs.append(100 * target_acc)
    num_runs = len(src_accs)
    mult = 1.645  # For 90% confidence intervals
    print("\nNon-adaptive accuracy on source (%): ", np.mean(src_accs),
          mult * np.std(src_accs) / np.sqrt(num_runs))
    print("Non-adaptive accuracy on target (%): ", np.mean(target_accs),
          mult * np.std(target_accs) / np.sqrt(num_runs))


def run_gradual_experiment(
    dataset_func, n_classes, input_shape, save_file, model_func=models.simple_softmax_conv_model,
    interval=2000, epochs=10, loss='ce', soft=False, conf_q=0.1, num_runs=20, num_repeats=None,
    run_all_self_train=True):
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
        gradual_accuracies, student, unsup_pseudolabels = utils.gradual_self_train(
            student_func, teacher, src_tr_x, src_tr_y, inter_x, inter_y, interval,
            epochs=epochs, soft=soft, confidence_q=conf_q)
        _, acc = student.evaluate(trg_eval_x, trg_eval_y)
        print("final gradual acc: ", acc)
        assert(inter_x.shape[0] == unsup_pseudolabels.shape[0])
        gradual_accuracies.append(acc)
        return src_acc, target_acc, gradual_accuracies
    results = []
    for i in range(num_runs):
        results.append(run(i))
    print('Saving to ' + save_file)
    pickle.dump(results, open(save_file, "wb"))


def gradual_experiment_results(save_name):
    results = pickle.load(open(save_name, "rb"))
    src_accs, target_accs = [], []
    final_graduals = []
    for src_acc, target_acc, gradual_accuracies in results:
        src_accs.append(100 * src_acc)
        target_accs.append(100 * target_acc)
        final_graduals.append(100 * gradual_accuracies[-1])
    num_runs = len(src_accs)
    mult = 1.645  # For 90% confidence intervals
    print("\nNon-adaptive accuracy on source (%): ", np.mean(src_accs),
          mult * np.std(src_accs) / np.sqrt(num_runs))
    print("Non-adaptive accuracy on target (%): ", np.mean(target_accs),
          mult * np.std(target_accs) / np.sqrt(num_runs))
    print("Gradual self-train accuracy (%): ", np.mean(final_graduals),
          mult * np.std(final_graduals) / np.sqrt(num_runs))


# Baseline experiment.
def portraits_source_conv_experiment():
    run_source_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/tune_st/source_portraits_no_aug.dat',
        model_func=models.simple_softmax_conv_model, epochs=20, loss='ce',
        num_runs=5, augmentation=False)


def portraits_source_conv_experiment_with_aug():
    run_source_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/tune_st/source_portraits_with_aug.dat',
        model_func=models.simple_softmax_conv_model, epochs=20, loss='ce',
        num_runs=5, augmentation=True)


if __name__ == "__main__":
    # portraits_source_conv_experiment()
    # print("Portraits source no aug")
    # source_experiment_results('saved_files/tune_st/source_portraits_no_aug.dat')
    portraits_source_conv_experiment_with_aug()
    print("Portraits source with aug")
    source_experiment_results('saved_files/tune_st/source_portraits_with_aug.dat')
