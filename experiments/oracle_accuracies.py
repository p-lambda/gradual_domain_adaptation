
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
        model = new_model()
        model.fit(inter_x[-interval:], inter_y[-interval:], epochs=epochs, verbose=True)
        _, target_acc = model.evaluate(trg_eval_x, trg_eval_y)
        return target_acc
    results = []
    for i in range(num_runs):
        results.append(run(i))
    print(np.mean(results))
    print(np.std(results) / np.sqrt(num_runs) * 1.645)
    print('Saving to ' + save_file)
    pickle.dump(results, open(save_file, "wb"))


def portraits_conv_oracle_experiment():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/oracle_portraits.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        num_runs=5)

if __name__ == "__main__":
    portraits_conv_oracle_experiment()

