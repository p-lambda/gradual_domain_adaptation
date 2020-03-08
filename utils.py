
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

def rand_seed(seed):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def self_train_once(student, teacher, unsup_x, confidence_q=0.1, epochs=20):
    # Do one bootstrapping step on unsup_x, where pred_model is used to make predictions,
    # and we use these predictions to update model.
    logits = teacher.predict(np.concatenate([unsup_x]))
    confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)
    alpha = np.quantile(confidence, confidence_q)
    indices = np.argwhere(confidence >= alpha)[:, 0]
    preds = np.argmax(logits, axis=1)
    student.fit(unsup_x[indices], preds[indices], epochs=epochs, verbose=False)


def soft_self_train_once(student, teacher, unsup_x, epochs=20):
    probs = teacher.predict(np.concatenate([unsup_x]))
    student.fit(unsup_x, probs, epochs=epochs, verbose=False)


def self_train(student_func, teacher, unsup_x, confidence_q=0.1, epochs=20, repeats=1,
               target_x=None, target_y=None, soft=False):
    accuracies = []
    for i in range(repeats):
        student = student_func(teacher)
        if soft:
            soft_self_train_once(student, teacher, unsup_x, epochs)
        else:
            self_train_once(student, teacher, unsup_x, confidence_q, epochs)
        if target_x is not None and target_y is not None:
            _, accuracy = student.evaluate(target_x, target_y, verbose=True)
            accuracies.append(accuracy)
        teacher = student
    return accuracies, student


def gradual_self_train(student_func, teacher, unsup_x, debug_y, interval, confidence_q=0.1,
                       epochs=20, soft=False):
    assert(not soft)
    upper_idx = int(unsup_x.shape[0] / interval)
    accuracies = []
    for i in range(upper_idx):
        student = student_func(teacher)
        cur_xs = unsup_x[interval*i:interval*(i+1)]
        cur_ys = debug_y[interval*i:interval*(i+1)]
        # _, student = self_train(
        #     student_func, teacher, unsup_x, confidence_q, epochs, repeats=2)
        if soft:
            soft_self_train_once(student, teacher, cur_xs, epochs)
        else:
            self_train_once(student, teacher, cur_xs, confidence_q, epochs)
        _, accuracy = student.evaluate(cur_xs, cur_ys)
        accuracies.append(accuracy)
        teacher_logits = teacher.predict(cur_xs)
        teacher_preds = np.argmax(teacher_logits, axis=1)
        student_logits = student.predict(cur_xs)
        student_preds = np.argmax(student_logits, axis=1)
        print('student-teacher agreement: ', np.mean(teacher_preds==student_preds))
        teacher = student
    return accuracies, student


def self_train_learn_gradual(student_func, teacher, unsup_x, debug_y, num_new_pts, epochs=20,
                             soft=False):
    num_unsup = unsup_x.shape[0]
    iters = int(num_unsup / num_new_pts)
    if iters * num_new_pts < num_unsup:
        iters += 1
    assert(iters * num_new_pts >= num_unsup)
    assert((iters-1) * num_new_pts < num_unsup)
    accuracies = []
    for i in range(iters):
        student = student_func(teacher)
        num_points_to_add = min((i+1) * num_new_pts, num_unsup)
        logits = teacher.predict(np.concatenate([unsup_x]))
        # TODO: change this to be top minus second top.
        confidence = np.amax(logits, axis=1)
        indices = np.argsort(confidence)
        # Plot average index as a function of confidence. Ideally this should be increasing.
        # averages = rolling_average(indices, num_new_pts)
        # plt.plot(list(range(len(averages))), averages)
        # plt.show()
        # Show histogram of angles for the top 2000 points.
        plot_histogram(indices[-num_points_to_add:] / 40000.0)
        indices = indices[-num_points_to_add:]
        preds = np.argmax(logits, axis=1)
        cur_xs = unsup_x[indices]
        cur_ys = debug_y[indices]
        pred_acc = np.mean(preds[indices] == cur_ys)
        print('accuracy: ', pred_acc)
        student.fit(cur_xs, preds[indices], epochs=epochs, verbose=False)
        _, accuracy = student.evaluate(cur_xs, cur_ys)
        accuracies.append(accuracy)
        teacher = student
    return accuracies, student


def split_data(xs, ys, splits):
    return np.split(xs, splits), np.split(ys, splits)


def train_to_acc(model, acc, train_x, train_y, val_x, val_y):
    # Modify steps per epoch to be around dataset size / 10
    # Keep training until accuracy 
    batch_size = 32
    data_size = train_x.shape[0]
    steps_per_epoch = int(data_size / 50.0 / batch_size)
    logger.info("train_xs size is %s", str(train_x.shape))
    while True:
        model.fit(train_x, train_y, batch_size=batch_size, steps_per_epoch=steps_per_epoch, verbose=False)
        val_accuracy = model.evaluate(val_x, val_y, verbose=False)[1]
        logger.info("validation accuracy is %f", val_accuracy)
        if val_accuracy >= acc:
            break
    return model


def save_model(model, filename):
    model.save(filename)


def load_model(filename):
    model = load_model(filename)


def rolling_average(sequence, r):
    N = sequence.shape[0]
    assert r < N
    assert r > 1
    rolling_sums = []
    cur_sum = sum(sequence[:r])
    rolling_sums.append(cur_sum)
    for i in range(r, N):
        cur_sum = cur_sum + sequence[i] - sequence[i-r]
        rolling_sums.append(cur_sum)
    return np.array(rolling_sums) * 1.0 / r


def plot_histogram(xs):
    bins = np.linspace(np.min(xs), np.max(xs), 40)
    plt.hist(xs, bins, alpha=0.5, label='hist')
    plt.show()

