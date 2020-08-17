
import gradual_st.utils as utils
import gradual_st.models as models
import gradual_st.datasets as datasets
import tensorflow as tf
from tensorflow.keras import metrics
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical


def compile_model(model, loss='ce'):
    loss = models.get_loss(loss, model.output_shape[1])
    model.compile(optimizer='adam',
                  loss=[loss],
                  metrics=['accuracy'])


def student_func_gen(model_func, retrain, loss):
    iters = 0
    def student_func(teacher):
        nonlocal iters
        iters += 1
        if iters == 1 or retrain:
            model = model_func()
            compile_model(model, loss)
            return model
        return teacher
    return student_func


def reg_vs_unreg_experiment(
    src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, trg_eval_x, trg_eval_y,
    n_classes, input_shape, save_file, unreg_model_func, reg_model_func,
    interval=2000, epochs=10, loss='ce', retrain=False, soft=False, num_runs=20):
    if soft:
        src_tr_y = to_categorical(src_tr_y)
        src_val_y = to_categorical(src_val_y)
        trg_eval_y = to_categorical(trg_eval_y)
        inter_y = to_categorical(inter_y)
    def teacher_model():
        model = unreg_model_func(n_classes, input_shape=input_shape)
        compile_model(model, loss)
        return model
    def run(seed):
        utils.rand_seed(seed)
        # Train source model.
        source_model = teacher_model()
        print(src_tr_x.shape, src_tr_y.shape)
        source_model.fit(src_tr_x, src_tr_y, epochs=epochs, verbose=False)
        _, src_acc = source_model.evaluate(src_val_x, src_val_y)
        _, target_acc = source_model.evaluate(trg_eval_x, trg_eval_y)
        # Gradual self-training.
        def gradual_train(regularize):
            teacher = teacher_model()
            teacher.set_weights(source_model.get_weights())
            if regularize:
                model_func = reg_model_func
            else:
                model_func = unreg_model_func
            student_func = student_func_gen(
                model_func=lambda: model_func(n_classes, input_shape=input_shape),
                retrain=retrain, loss=loss)
            accuracies, student, _ = utils.gradual_self_train(
                student_func, teacher, src_tr_x, src_tr_y, inter_x, inter_y, interval, epochs=epochs, soft=soft)
            _, acc = student.evaluate(trg_eval_x, trg_eval_y)
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
    print('Saving to ' + save_file)
    pickle.dump(results, open(save_file, "wb"))


def rotated_mnist_regularization_experiment(
    unreg_model_func, reg_model_func, loss, save_name_base, N=2000, delta_angle=3, num_angles=21,
    retrain=False, num_runs=20):
    # Get data.
    utils.rand_seed(0)
    (train_x, train_y), _ = datasets.get_preprocessed_mnist()
    orig_x, orig_y = train_x[:N], train_y[:N]
    inter_x, inter_y = datasets.make_population_rotated_dataset(
        orig_x, orig_y, delta_angle, num_angles)
    trg_x, trg_y = inter_x[-N:], inter_y[-N:]
    n_classes = 10
    input_shape = (28, 28, 1)
    loss = 'ce'
    epochs = 20
    interval = N
    save_file = (save_name_base + '_' + str(N) + '_' + str(delta_angle) + '_' + str(num_angles) +
                 '.dat')
    reg_vs_unreg_experiment(
        orig_x, orig_y, orig_x, orig_y, inter_x, inter_y, trg_x, trg_y,
        n_classes, input_shape, save_file, unreg_model_func, reg_model_func,
        interval, epochs, loss, retrain, soft=False, num_runs=num_runs)


def finite_data_experiment(
    dataset_func, n_classes, input_shape, save_file, unreg_model_func, reg_model_func,
    interval=2000, epochs=10, loss='ce', retrain=False, soft=False, num_runs=20):
    utils.rand_seed(0)
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()
    reg_vs_unreg_experiment(
        src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, trg_val_x, trg_val_y,
        n_classes, input_shape, save_file, unreg_model_func, reg_model_func,
        interval, epochs, loss, retrain, soft=soft, num_runs=num_runs)


def regularization_results(save_name):
    results = pickle.load(open(save_name, "rb"))
    src_accs, target_accs, reg_accs, unreg_accs = [], [], [], []
    for src_acc, target_acc, reg_accuracies, unreg_accuracies in results:
        src_accs.append(100 * src_acc)
        target_accs.append(100 * target_acc)
        reg_accs.append(100 * reg_accuracies[-1])
        unreg_accs.append(100 * unreg_accuracies[-1])
    num_runs = len(src_accs)
    mult = 1.645  # For 90% confidence intervals
    print("\nNon-adaptive accuracy on source (%): ", np.mean(src_accs),
          mult * np.std(src_accs) / np.sqrt(num_runs))
    print("Non-adaptive accuracy on target (%): ", np.mean(target_accs),
          mult * np.std(target_accs) / np.sqrt(num_runs))
    print("Reg accuracy (%): ", np.mean(reg_accs),
          mult * np.std(reg_accs) / np.sqrt(num_runs))
    print("Unreg accuracy (%): ", np.mean(unreg_accs),
          mult * np.std(unreg_accs) / np.sqrt(num_runs))


def rotated_mnist_60_conv_experiment():
    finite_data_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/reg_vs_unreg_rot_mnist_60_conv.dat',
        unreg_model_func=models.unregularized_softmax_conv_model,
        reg_model_func=models.simple_softmax_conv_model,
        interval=2000, epochs=10, loss='ce', soft=False, num_runs=5)


def soft_rotated_mnist_60_conv_experiment():
    finite_data_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/reg_vs_unreg_soft_rot_mnist_60_conv.dat',
        unreg_model_func=models.unregularized_softmax_conv_model,
        reg_model_func=models.simple_softmax_conv_model,
        interval=2000, epochs=10, loss='categorical_ce', soft=True, num_runs=5)


def retrain_soft_rotated_mnist_60_conv_experiment():
    finite_data_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/reg_vs_unreg_retrain_soft_rot_mnist_60_conv.dat',
        unreg_model_func=models.unregularized_softmax_conv_model,
        reg_model_func=models.simple_softmax_conv_model,
        interval=2000, epochs=10, loss='categorical_ce', soft=True, num_runs=5)


def keras_retrain_soft_rotated_mnist_60_conv_experiment():
    finite_data_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/reg_vs_unreg_keras_retrain_soft_rot_mnist_60_conv.dat',
        unreg_model_func=models.unregularized_keras_mnist_model,
        reg_model_func=models.keras_mnist_model,
        interval=2000, epochs=10, loss='categorical_ce', soft=True, num_runs=5)


def deeper_retrain_soft_rotated_mnist_60_conv_experiment():
    finite_data_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/deeper_retrain_soft_rot_mnist_60_conv.dat',
        unreg_model_func=models.deeper_softmax_conv_model,
        reg_model_func=models.deeper_softmax_conv_model,
        interval=2000, epochs=10, loss='categorical_ce', soft=True, num_runs=5)


def cov_small_mlp_experiment():
    finite_data_experiment(
        dataset_func=datasets.cov_data_small_func, n_classes=2, input_shape=(54,),
        save_file='saved_files/reg_vs_unreg_covtype_small.dat',
        unreg_model_func=models.unregularized_mlp_softmax_model,
        reg_model_func=models.mlp_softmax_model, interval=50000, epochs=5, loss='ce',
        soft=False, num_runs=5)


def soft_cov_small_mlp_experiment():
    finite_data_experiment(
        dataset_func=datasets.cov_data_small_func, n_classes=2, input_shape=(54,),
        save_file='saved_files/reg_vs_unreg_soft_covtype_small.dat',
        unreg_model_func=models.unregularized_mlp_softmax_model,
        reg_model_func=models.mlp_softmax_model, interval=50000, epochs=5, loss='categorical_ce',
        soft=True, num_runs=5)


def portraits_conv_experiment():
    finite_data_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/reg_vs_unreg_portraits.dat',
        unreg_model_func=models.unregularized_softmax_conv_model,
        reg_model_func=models.simple_softmax_conv_model,
        interval=2000, epochs=20, loss='ce', soft=False, num_runs=5)


def soft_portraits_conv_experiment():
    finite_data_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/reg_vs_unreg_soft_portraits.dat',
        unreg_model_func=models.unregularized_softmax_conv_model,
        reg_model_func=models.simple_softmax_conv_model,
        interval=2000, epochs=20, loss='categorical_ce', soft=True, num_runs=5)


def gaussian_data_func(d):
    return datasets.make_high_d_gaussian_data(
        d=d, min_var=0.05, max_var=0.1,
        source_alphas=[0.0, 0.0], inter_alphas=[0.0, 1.0], target_alphas=[1.0, 1.0],
        n_src_tr=500, n_src_val=1000, n_inter=5000, n_trg_val=1000, n_trg_tst=1000)


def gaussian_linear_experiment():
    d = 100        
    finite_data_experiment(
        dataset_func=lambda: gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/reg_vs_unreg_gaussian.dat',
        unreg_model_func=lambda k, input_shape: models.linear_softmax_model(k, input_shape, l2_reg=0.0),
        reg_model_func=models.linear_softmax_model,
        interval=500, epochs=100, loss='ce', soft=False, num_runs=5)

def soft_gaussian_linear_experiment():
    d = 100        
    finite_data_experiment(
        dataset_func=lambda: gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/reg_vs_unreg_soft_gaussian.dat',
        unreg_model_func=lambda k, input_shape: models.linear_softmax_model(k, input_shape, l2_reg=0.0),
        reg_model_func=models.linear_softmax_model,
        interval=500, epochs=100, loss='categorical_ce', soft=True, num_runs=5)


def dialing_rotated_mnist_60_conv_experiment():
    finite_data_experiment(
        dataset_func=datasets.rotated_mnist_60_dialing_ratios_data_func, n_classes=10,
        input_shape=(28, 28, 1),
        save_file='saved_files/reg_vs_unreg_dialing_rot_mnist_60_conv.dat',
        unreg_model_func=models.simple_softmax_conv_model,
        reg_model_func=models.simple_softmax_conv_model,
        interval=2000, epochs=10, loss='ce', soft=False, num_runs=5)


if __name__ == "__main__":
    # rotated_mnist_regularization_experiment(
    #     models.unregularized_softmax_conv_model, models.simple_softmax_conv_model, 'ce',
    #     save_name_base='saved_files/inf_reg_mnist', N=2000, delta_angle=3, num_angles=20,
    #     retrain=False, num_runs=5)
    # print("Rot MNIST experiment 2000 points rotated")
    # regularization_results('saved_files/inf_reg_mnist_2000_3_20.dat')

    # rotated_mnist_regularization_experiment(
    #     models.unregularized_softmax_conv_model, models.simple_softmax_conv_model, 'ce',
    #     save_name_base='saved_files/inf_reg_mnist', N=5000, delta_angle=3, num_angles=20,
    #     retrain=False, num_runs=5)
    # print("Rot MNIST experiment 5000 points rotated")
    # regularization_results('saved_files/inf_reg_mnist_5000_3_20.dat')

    # rotated_mnist_regularization_experiment(
    #     models.unregularized_softmax_conv_model, models.simple_softmax_conv_model, 'ce',
    #     save_name_base='saved_files/inf_reg_mnist', N=20000, delta_angle=3, num_angles=20,
    #     retrain=False, num_runs=5)
    # print("Rot MNIST experiment 20k points rotated")
    # regularization_results('saved_files/inf_reg_mnist_20000_3_20.dat')

    # Run all experiments comparing regularization vs no regularization.
    # cov_small_mlp_experiment()
    print("Cover Type experiment reg vs no reg")
    regularization_results('saved_files/reg_vs_unreg_covtype_small.dat')
    # portraits_conv_experiment()
    # print("Portraits conv experiment reg vs no reg")
    # regularization_results('saved_files/reg_vs_unreg_portraits.dat')
    # rotated_mnist_60_conv_experiment()
    # print("Rotating MNIST conv experiment reg vs no reg")
    # regularization_results('saved_files/reg_vs_unreg_rot_mnist_60_conv.dat')
    # gaussian_linear_experiment()
    # print("Gaussian linear experiment reg vs no reg")
    # regularization_results('saved_files/reg_vs_unreg_gaussian.dat')

    # Run all experiments, soft labeling, comparing regularization vs no regularization.
    soft_cov_small_mlp_experiment()
    print("Cover Type experiment soft labeling reg vs no reg")
    regularization_results('saved_files/reg_vs_unreg_soft_covtype_small.dat')
    # soft_portraits_conv_experiment()
    # print("Portraits conv experiment soft labeling reg vs no reg")
    # regularization_results('saved_files/reg_vs_unreg_soft_portraits.dat')
    # soft_rotated_mnist_60_conv_experiment()
    # print("Rot MNIST conv experiment soft labeling reg vs no reg")
    # regularization_results('saved_files/reg_vs_unreg_soft_rot_mnist_60_conv.dat')
    # soft_gaussian_linear_experiment()
    # print("Gaussian linear experiment soft labeling reg vs no reg")
    # regularization_results('saved_files/reg_vs_unreg_soft_gaussian.dat')

    # # Dialing ratios results.
    # dialing_rotated_mnist_60_conv_experiment()
    # print("Dialing rations MNIST experiment reg vs no reg")
    # regularization_results('saved_files/reg_vs_unreg_dialing_rot_mnist_60_conv.dat')

    # # Try retraining the model each iteration.
    # retrain_soft_rotated_mnist_60_conv_experiment()
    # print("Rot MNIST conv experiment reset model when self-training")
    # regularization_results('saved_files/reg_vs_unreg_retrain_soft_rot_mnist_60_conv.dat')
    # # Use the Keras MNIST model.
    # keras_retrain_soft_rotated_mnist_60_conv_experiment()
    # print("Rot MNIST conv experiment use Keras MNIST model")
    # regularization_results('saved_files/reg_vs_unreg_keras_retrain_soft_rot_mnist_60_conv.dat')
    # # Use a deeper (4 layer) conv net model.
    # deeper_retrain_soft_rotated_mnist_60_conv_experiment()
    # print("Rot MNIST conv experiment use deeper model")
    # regularization_results('saved_files/deeper_retrain_soft_rot_mnist_60_conv.dat')

