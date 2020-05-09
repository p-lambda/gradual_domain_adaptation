
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


def gradual_self_train(student_func, teacher, src_tr_x, src_tr_y, unsup_x, debug_y, interval,
                       confidence_q=0.1, epochs=20, soft=False, accumulate=False):
    assert(not soft)
    upper_idx = int(unsup_x.shape[0] / interval)
    accuracies = []
    unsup_pseudolabels = []
    for i in range(upper_idx):
        student = student_func(teacher)
        # Have an option to accumulate instead of just going to the next interval.
        if accumulate:
            cur_xs = np.concatenate([src_tr_x, unsup_x[:interval*(i+1)]])
            cur_ys = np.concatenate([src_tr_y, debug_y[:interval*(i+1)]])
        else:
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
        if not accumulate:
            unsup_pseudolabels.append(student_preds)
        teacher = student
    # Print average, min, max student teacher agreement
    # Plot average accuracy on entire unsup set and current set.
    # Current set is just the most recent set of points.
    if not accumulate:
        unsup_pseudolabels = np.concatenate(unsup_pseudolabels)
    return accuracies, student, unsup_pseudolabels


def self_train_learn_gradual(student_func, teacher, src_tr_x, src_tr_y, unsup_x, debug_y,
                             num_new_pts, save_folder, seed, epochs=20, soft=False, use_src=True,
                             accumulate=True, conf_stop=1.0):
    # TODO: deal with accumulation.
    num_unsup = unsup_x.shape[0]
    iters = int(num_unsup / num_new_pts)
    if iters * num_new_pts < num_unsup:
        iters += 1
    assert(iters * num_new_pts >= num_unsup)
    assert((iters-1) * num_new_pts < num_unsup)
    cur_accuracies = []
    all_accuracies = []
    pseudo_losses = []
    disagreements = []
    accumulated_xs = []
    accumulated_pseudo = []
    for i in range(iters):
        student = student_func(teacher)
        logits = teacher.predict(unsup_x)
        # TODO: maybe change this to be top minus second top.
        confidence = np.amax(logits, axis=1)
        indices = np.argsort(confidence)
        if accumulate:
            num_points_to_add = min((i+1) * num_new_pts, num_unsup)
            select_indices = indices[-num_points_to_add:]
        else:
            assert conf_stop == 1.0
            num_points_to_add = num_new_pts
            select_indices = indices[-num_points_to_add:]
            print("selected indices shape", select_indices.shape)
        # Don't train on predictions the model is "too" confident about. We don't want to overtrain.
        # But we don't want to filter out pretty much all the examples with this filtering process.
        print("Filtering out # examples: ", np.sum(confidence > conf_stop))
        print(np.max(confidence))
        too_conf_count = np.minimum(select_indices.shape[0] / 2, np.sum(confidence > conf_stop))
        too_conf_count = int(too_conf_count)
        if too_conf_count > 0:
            select_indices = select_indices[:-too_conf_count]
        # Plot average index as a function of confidence. Ideally this should be increasing.
        # averages = rolling_average(indices, num_new_pts)
        # plt.plot(list(range(len(averages))), averages)
        # plt.show()
        # Show histogram of angles for the points to add.
        # plot_histogram(indices[-num_points_to_add:] / 40000.0)
        teacher_preds = np.argmax(logits, axis=1)
        cur_xs = unsup_x[select_indices]
        cur_ys = debug_y[select_indices]
        pseudo_ys = teacher_preds[select_indices]
        cur_acc = np.mean(pseudo_ys == cur_ys)
        cur_accuracies.append(cur_acc)
        print('accuracy: ', cur_acc)
        all_acc = np.mean(teacher_preds == debug_y)
        all_accuracies.append(all_acc)
        print('all acc: ', all_acc)
        if use_src:
            student.fit(np.concatenate([src_tr_x, cur_xs], axis=0),
                        np.concatenate([src_tr_y, pseudo_ys], axis=0),
                        epochs=epochs, verbose=False)
        else:
            student.fit(cur_xs, pseudo_ys, epochs=epochs, verbose=False)
        pseudo_loss, _ = student.evaluate(cur_xs, pseudo_ys)
        pseudo_losses.append(pseudo_loss)
        student_preds = np.argmax(student.predict(cur_xs), axis=1)
        student_teacher_disagreement = np.mean(student_preds != pseudo_ys)
        disagreements.append(student_teacher_disagreement)
        if not accumulate:
            unsup_x = unsup_x[indices[:-num_points_to_add]]
            print("unsup data shape", unsup_x.shape)
            debug_y = debug_y[indices[:-num_points_to_add]]
            accumulated_xs.append(cur_xs)
            accumulated_pseudo.append(pseudo_ys)
        teacher = student
    disagreement_summary = (np.min(disagreements), np.mean(disagreements),
                            np.max(disagreements))
    print('teacher disagreements: ', disagreement_summary)
    # Save accuracy and loss plots.
    save_name = save_folder + '/all_accs_' + str(seed)
    save_plot(save_name, all_accuracies, x_label='self-training iters', y_label='all acc')
    save_name = save_folder + '/pseudo_losses_' + str(seed)
    save_plot(save_name, pseudo_losses, x_label='self-training iters', y_label='pseudo losses')
    save_name = save_folder + '/cur_accs_' + str(seed)
    save_plot(save_name, cur_accuracies, x_label='self-training iters', y_label='cur acc')
    if not accumulate:
        accumulated_xs.append(src_tr_x)
        accumulated_pseudo.append(src_tr_y)
        accumulated_xs = np.concatenate(accumulated_xs)
        accumulated_pseudo = np.concatenate(accumulated_pseudo)
    return disagreement_summary, all_accuracies, pseudo_losses, cur_accuracies, student, accumulated_xs, accumulated_pseudo


def safe_ceil_divide(a, b):
    # a and b should be positive integers
    assert a > 0 and b > 0
    r = int(a / b)
    if b * r < a:
        r += 1
    assert(r * b >= a)
    assert((r-1) * b < a)
    return r


def split(sequence, parts):
    assert parts <= len(sequence)
    part_size = int(np.ceil(len(sequence) * 1.0 / parts))
    assert part_size * parts >= len(sequence)
    assert (part_size - 1) * parts < len(sequence)
    return [sequence[i:i + part_size] for i in range(0, len(sequence), part_size)]


def check_split_shapes(splitted, original):
    assert type(splitted) == list
    assert len(splitted[0].shape) == len(original.shape)
    assert sum([xs.shape[0] for xs in splitted]) == original.shape[0]


def get_mean_group_confidences(confidences, num_groups):
    grouped_confidences = split(confidences, num_groups)
    check_split_shapes(grouped_confidences, confidences)
    assert len(grouped_confidences[0].shape) == 1
    mean_confidences = [np.mean(c) for c in grouped_confidences]
    return mean_confidences


def get_most_confident_group_indices(group_confidences, num_groups_to_add, avoid_groups_set):
    assert num_groups_to_add > 0
    group_confidences = np.array(group_confidences)
    num_groups_to_add = min(num_groups_to_add, len(group_confidences))
    assert(np.min(group_confidences) >= 0.0)
    group_confidences[list(avoid_groups_set)] = -1.0
    indices = np.argsort(group_confidences)
    selected_groups = indices[-num_groups_to_add:]
    assert(len(avoid_groups_set.intersection(set(list(selected_groups)))) == 0)
    return selected_groups


def get_selected_indices(num_pts, num_groups, selected_groups):
    grouped_indices = split(list(range(num_pts)), num_groups)
    assert type(grouped_indices[0]) == list
    grouped_selected_indices = [grouped_indices[g] for g in selected_groups]
    flatten = lambda l: [item for sublist in l for item in sublist]
    selected_indices = flatten(grouped_selected_indices)
    assert type(selected_indices) == list
    assert type(selected_indices[0]) == int
    group_size = int(np.ceil(num_pts * 1.0 / num_groups))
    assert len(selected_indices) <= group_size * len(selected_groups)
    assert len(selected_indices) > group_size * (len(selected_groups) - 1)
    return selected_indices


def self_train_from_pseudolabels(student, src_tr_x, src_tr_y, unsup_x, teacher_preds,
                                 selected_indices, epochs):
    cur_xs = unsup_x[selected_indices]
    cur_teacher_preds = teacher_preds[selected_indices]
    student.fit(np.concatenate([src_tr_x, cur_xs], axis=0),
                np.concatenate([src_tr_y, cur_teacher_preds], axis=0),
                epochs=epochs, verbose=False)


def get_stats(student, unsup_x, teacher_preds, debug_y, selected_indices):
    cur_xs = unsup_x[selected_indices]
    cur_teacher_preds = teacher_preds[selected_indices]
    cur_ys = debug_y[selected_indices]
    cur_acc = np.mean(cur_ys == cur_teacher_preds)
    all_acc = np.mean(teacher_preds == debug_y)
    student_preds = np.argmax(student.predict(cur_xs), axis=1)
    student_teacher_disagreement = np.mean(student_preds != cur_teacher_preds)
    pseudo_loss, _ = student.evaluate(cur_xs, cur_teacher_preds, verbose=False)
    return all_acc, cur_acc, student_teacher_disagreement, pseudo_loss


def print_stats(stats):
    all_acc, cur_acc, student_teacher_disagreement, pseudo_loss = stats
    print('{:<10} {:<10} {:<15} {}'.format('All Acc', 'Cur Acc', 'Stu-Tea-Dis', 'Pseudo-loss'))
    print('{:<10.2f} {:<10.2f} {:<15.2f} {:.2f}'.format(
        all_acc * 100, cur_acc * 100, student_teacher_disagreement * 100, pseudo_loss * 100))


def save_stats(save_folder, seed, stats_list, accumulate):
    print("Save function to be implemented.")


def get_quantiles(values, num_groups):
    values = values / float(num_groups)
    return (np.quantile(values, 0.1), np.quantile(values, 0.25), np.quantile(values, 0.5),
        np.quantile(values, 0.75), np.quantile(values, 0.9))


def self_train_learn_gradual_group(student_func, teacher, src_tr_x, src_tr_y, unsup_x, debug_y, num_groups,
                                   num_new_groups, save_folder, seed, epochs=20, accumulate=True,
                                   confidence_scorer=None):
    assert num_new_groups < num_groups
    num_iters = safe_ceil_divide(num_groups, num_new_groups)
    stats_list = []
    used_groups = set()
    accumulated_xs = []
    accumulated_pseudos = []
    if confidence_scorer is not None:
            confidences = confidence_scorer.predict(unsup_x)[:, 0]
    for i in range(num_iters):
        student = student_func(teacher)
        logits = teacher.predict(unsup_x)
        if confidence_scorer is None:
            confidences = np.amax(logits, axis=1)
        teacher_preds = np.argmax(logits, axis=1)
        mean_group_confidences = get_mean_group_confidences(confidences, num_groups)
        confidence_ranks_by_time = get_confidence_ranks_by_time(mean_group_confidences)
        # plot_confidence_ranks_by_time(confidence_ranks_by_time)
        if accumulate:
            selected_groups = get_most_confident_group_indices(
                mean_group_confidences, (i+1) * num_new_groups, avoid_groups_set=used_groups)
        else:
            selected_groups = get_most_confident_group_indices(
                mean_group_confidences, num_new_groups, avoid_groups_set=used_groups)
            print(selected_groups)
        interquartiles = get_quantiles(selected_groups, num_groups)
        print("Quantiles of selected groups: ", interquartiles)
        selected_indices = get_selected_indices(len(unsup_x), num_groups, selected_groups)
        # Self-training only uses unlabeled intermediate data and pseudolabels.
        self_train_from_pseudolabels(student, src_tr_x, src_tr_y, unsup_x, teacher_preds,
                                     selected_indices, epochs)
        if not accumulate:
            used_groups.update(selected_groups)
            accumulated_xs.append(unsup_x[selected_indices])
            accumulated_pseudos.append(teacher_preds[selected_indices])
        # Get statistics that we can output and save to a file.
        stats = get_stats(student, unsup_x, teacher_preds, debug_y, selected_indices)
        print_stats(stats)
        stats_list.append(stats)
        teacher = student
    save_stats(save_folder, seed, stats_list, accumulate)
    if not accumulate:
        accumulated_xs = np.concatenate(accumulated_xs)
        accumulated_pseudos = np.concatenate(accumulated_pseudos)
    return stats, student, accumulated_xs, accumulated_pseudos


def save_plot(save_name, values, x_label, y_label):
    plt.clf()
    plt.plot(list(range(len(values))), values)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.savefig(save_name)


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


def invert(xs):
    # xs should be a sorted list of 0, ..., n-1, unque indices, permuted in some way
    assert len(xs.shape) == 1
    inverted_xs = np.zeros(len(xs))
    inverted_xs[xs] = np.arange(len(xs))
    return inverted_xs


def get_confidence_ranks_by_time(confidences):
    return invert(np.argsort(confidences))


def plot_confidence_ranks_by_time(confidence_ranks_by_time):
    confidence_ranks_by_time = confidence_ranks_by_time / (len(confidence_ranks_by_time) - 1.0)
    plt.clf()
    plt.plot(np.arange(len(confidence_ranks_by_time)), confidence_ranks_by_time)
    plt.show()
