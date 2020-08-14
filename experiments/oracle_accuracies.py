
import gradual_st.utils as utils
import gradual_st.models as models
import gradual_st.datasets as datasets
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
import pickle


def compile_model(model, loss='ce'):
    loss = models.get_loss(loss, model.output_shape[1])
    model.compile(optimizer='adam',
                  loss=[loss],
                  metrics=[metrics.sparse_categorical_accuracy])

def run_experiment(
    dataset_func, n_classes, input_shape, save_file, model_func=models.simple_softmax_conv_model,
    interval=2000, epochs=10, loss='ce', num_runs=20):
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
        source_model = new_model()
        source_model.fit(src_tr_x, src_tr_y, epochs=epochs, verbose=False)
        _, target_acc = source_model.evaluate(trg_eval_x, trg_eval_y)
        oracle_model = new_model()
        oracle_model.fit(inter_x[-interval:], inter_y[-interval:], epochs=epochs, verbose=False)
        _, oracle_acc = oracle_model.evaluate(trg_eval_x, trg_eval_y)
        return target_acc, oracle_acc
    results = []
    for i in range(num_runs):
        results.append(run(i))
    print(np.mean(results))
    print(np.std(results) / np.sqrt(num_runs) * 1.645)
    print('Saving to ' + save_file)
    pickle.dump(results, open(save_file, "wb"))


def experiment_results(save_name):
    results = pickle.load(open(save_name, "rb"))
    target_accs = 100 * np.array(results)[:, 0]
    oracle_accs = 100 * np.array(results)[:, 1]
    oracle_accs_extra = oracle_accs - target_accs
    num_runs = len(target_accs)
    mult = 1.645  # For 90% confidence intervals
    print("Non-adaptive accuracy on target (%): ", np.mean(target_accs),
          mult * np.std(target_accs) / np.sqrt(num_runs))
    print("Oracle accuracy (%): +", np.mean(oracle_accs_extra),
          mult * np.std(oracle_accs_extra) / np.sqrt(num_runs))


def portraits_conv_oracle_experiment():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/oracle_portraits.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        num_runs=5)


def cov_small_mlp_experiment():
    run_experiment(
        dataset_func=datasets.cov_data_small_func, n_classes=2, input_shape=(54,),
        save_file='saved_files/oracle_covtype_small.dat',
        model_func=models.mlp_softmax_model, interval=50000, epochs=5, loss='ce',
        num_runs=5)


def rotated_mnist_60_conv_experiment():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/oracle_rot_mnist_60_conv.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        num_runs=5)


if __name__ == "__main__":
    print("Cov Type")
    # cov_small_mlp_experiment()
    experiment_results('saved_files/oracle_covtype_small.dat')
    print("Rotating MNIST")
    # rotated_mnist_60_conv_experiment()
    experiment_results('saved_files/oracle_rot_mnist_60_conv.dat')
    print("Portraits")
    # portraits_conv_oracle_experiment()
    experiment_results('saved_files/oracle_portraits.dat')

