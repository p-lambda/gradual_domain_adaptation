import scipy.io as sio
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy.io
from scipy import ndimage
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import regularizers
import collections
import nearest_neighbor


image_options = {
	'target_size': (32, 32),
	'batch_size': 100,
	'class_mode': 'binary',
	'color_mode': 'grayscale',
}


def rand_seed(seed):
	np.random.seed(seed)
	tf.compat.v1.set_random_seed(seed)


def linear_model(num_labels, input_shape, l2_reg=0.02):
    linear_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(num_labels, activation=None, name='out',
    	kernel_regularizer=regularizers.l2(l2_reg))
    ])
    return linear_model


def simple_conv_model(num_labels, hidden_nodes=32, input_shape=(28,28,1), l2_reg=0.0):
    return keras.models.Sequential([
    keras.layers.Conv2D(hidden_nodes, (5,5), (2,2), activation=tf.nn.relu,
                           padding='same', input_shape=input_shape),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2,2), activation=tf.nn.relu,
                           padding='same'),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2,2), activation=tf.nn.relu,
                           padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(name='after_flatten'),
    # keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
    ])


def papernot_model(num_labels, input_shape=(28,28,1)):
    papernot_conv_model = keras.models.Sequential([
     keras.layers.Conv2D(64, (8, 8), (2,2), activation=tf.nn.relu,
                            padding='same', input_shape=input_shape),
     keras.layers.Conv2D(128, (6,6), (2,2), activation=tf.nn.relu,
                            padding='valid'),
     keras.layers.Conv2D(128, (5,5), (1,1), activation=tf.nn.relu,
                            padding='valid'),
     keras.layers.Flatten(name='after_flatten'),
     keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
    ])
    return papernot_conv_model


def save_data(dir='dataset_32x32', save_file='dataset_32x32.mat'):
	Xs, Ys = [], []
	datagen = ImageDataGenerator(rescale=1./255)
	data_generator = datagen.flow_from_directory(
		'dataset_32x32/', shuffle=False, **image_options)
	while True:
		next_x, next_y = data_generator.next()
		Xs.append(next_x)
		Ys.append(next_y)
		if data_generator.batch_index == 0:
			break
	Xs = np.concatenate(Xs)
	Ys = np.concatenate(Ys)
	filenames = [f[2:] for f in data_generator.filenames]
	filenames_idx = list(zip(filenames, range(len(filenames))))
	filenames_idx = [(f, i) for f, i in zip(filenames, range(len(filenames)))]
	                 # if f[5:8] == 'Cal' or f[5:8] == 'cal']
	indices = [i for f, i in sorted(filenames_idx)]
	# print(filenames)
	# sort_indices = np.argsort(filenames)
	# We need to sort only by year, and not have correlation with state.
	# print state stats? print gender stats? print school stats?
	# E.g. if this changes a lot by year, then we might want to do some grouping.
	# Maybe print out number per year, and then we can decide on a grouping? Or algorithmically decide?
	Xs = Xs[indices]
	Ys = Ys[indices]
	scipy.io.savemat('./' + save_file, mdict={'Xs': Xs, 'Ys': Ys})


def load_faces_data(load_file='dataset_32x32.mat'):
	data = scipy.io.loadmat('./' + load_file)
	return data['Xs'], data['Ys'][0]


def split_sizes(array, sizes):
	indices = np.cumsum(sizes)
	return np.split(array, indices)


def shuffle(xs, ys):
	indices = list(range(len(xs)))
	np.random.shuffle(indices)
	return xs[indices], ys[indices]


# Synthetic Datasets.

def make_rotated_mnist(start_angle, end_angle, num_points):
    images, labels = [], []
    (train_x, train_y), (_, _) = mnist.load_data()
    train_x = train_x / 255.0
    for i in range(num_points):
    	angle = float(end_angle - start_angle) / num_points * i + start_angle
    	idx = np.random.choice(train_x.shape[0], 1)[0]
    	img = ndimage.rotate(train_x[idx], angle, reshape=False)
    	images.append(img)
    	labels.append(train_y[idx])
    Xs = np.expand_dims(np.array(images), axis=-1)
    return Xs, np.array(labels)


def make_moving_gaussians(source_means, source_sigmas, target_means, target_sigmas, steps):
	def shape_means(means):
		means = np.array(means)
		if len(means.shape) == 1:
			means = np.expand_dims(means, axis=-1)
		else:
			assert(len(means.shape) == 2)
		return means
	def shape_sigmas(sigmas, means):
		sigmas = np.array(sigmas)
		shape_len = len(sigmas.shape)
		assert(shape_len == 1 or shape_len == 3)
		if shape_len == 1:
			c = np.expand_dims(np.expand_dims(sigmas, axis=-1), axis=-1)
			d = means.shape[1]
			new_sigmas = c * np.eye(d)
			assert(new_sigmas.shape == (sigmas.shape[0], d, d))
		return new_sigmas
	source_means = shape_means(source_means)
	target_means = shape_means(target_means)
	source_sigmas = shape_sigmas(source_sigmas, source_means)
	target_sigmas = shape_sigmas(target_sigmas, target_means)
	num_classes = source_means.shape[0]
	class_prob = 1.0 / num_classes
	xs = []
	ys = []
	for i in range(steps):
		y = np.argmax(np.random.multinomial(1, [class_prob] * num_classes))
		alpha = float(i) / (steps - 1)
		mean = source_means[y] * (1 - alpha) + target_means[y] * alpha
		sigma = source_sigmas[y] * (1 - alpha) + target_sigmas[y] * alpha
		x = np.random.multivariate_normal(mean, sigma)
		xs.append(x)
		ys.append(y)
	return np.array(xs), np.array(ys)


def high_d_gaussians(d, var, n):
	# Choose random direction.
	v = np.random.multivariate_normal(np.zeros(d), np.eye(d))
	v = v / np.linalg.norm(v)
	# Choose random perpendicular direction.
	perp = np.random.multivariate_normal(np.zeros(d), np.eye(d))
	perp = perp - np.dot(perp, v) * v
	perp = perp / np.linalg.norm(perp)
	assert(abs(np.dot(perp, v)) < 1e-8)
	assert(abs(np.linalg.norm(v) - 1) < 1e-8)
	assert(abs(np.linalg.norm(perp) - 1) < 1e-8)
	s_a = 2 * perp - v
	s_b = -2 * perp + v
	t_a = -2 * perp - v
	t_b = 2 * perp + v
	return lambda: make_moving_gaussians([s_a, s_b], [var, var], [t_a, t_b], [var, var], n)


def get_split_data(dataset):
	Xs, Ys = dataset.get_data()
	n_src = dataset.n_src_train + dataset.n_src_valid
	n_target = dataset.n_target_unsup + dataset.n_target_val + dataset.n_target_test
	src_x, src_y = shuffle(Xs[:n_src], Ys[:n_src])
	target_x, target_y = shuffle(
		Xs[dataset.target_end-n_target:dataset.target_end],
		Ys[dataset.target_end-n_target:dataset.target_end])
	[src_train_x, src_val_x] = split_sizes(src_x, [dataset.n_src_train])
	[src_train_y, src_val_y] = split_sizes(src_y, [dataset.n_src_train])
	[target_unsup_x, target_val_x, final_target_test_x] = split_sizes(
		target_x, [dataset.n_target_unsup, dataset.n_target_val])
	[debug_target_unsup_y, target_val_y, final_target_test_y] = split_sizes(
		target_y, [dataset.n_target_unsup, dataset.n_target_val])
	inter_x, inter_y = Xs[n_src:dataset.target_end-n_target], Ys[n_src:dataset.target_end-n_target]
	return SplitData(
		src_train_x=src_train_x,
		src_val_x=src_val_x,
		src_train_y=src_train_y,
		src_val_y=src_val_y,
		target_unsup_x=target_unsup_x,
		target_val_x=target_val_x,
		final_target_test_x=final_target_test_x,
	 	debug_target_unsup_y=debug_target_unsup_y,
	 	target_val_y=target_val_y,
	 	final_target_test_y=final_target_test_y,
	 	inter_x=inter_x,
	 	inter_y=inter_y,
	)


def sparse_categorical_hinge(num_classes):
	def loss(y_true,y_pred):
		y_true = tf.reduce_mean(y_true, axis=1)
		y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=num_classes)
		return losses.categorical_hinge(y_true, y_pred)
	return loss


def run_model_on_train(model, split_data, train_x, train_y, epochs=1000, loss='hinge'):
	if loss == 'hinge':
		loss = sparse_categorical_hinge(model.output_shape[1])
	else:
		loss = losses.sparse_categorical_crossentropy
	model.compile(optimizer='adam',
	              loss=[loss],
	              metrics=[metrics.sparse_categorical_accuracy])
	model.fit(train_x, train_y, epochs=epochs, verbose=False)
	print("Source accuracy:")
	model.evaluate(split_data.src_val_x, split_data.src_val_y)
	print("Target accuracy:")
	model.evaluate(split_data.target_val_x, split_data.target_val_y)


def run_model(model, split_data, epochs=1000, loss='hinge'):
	run_model_on_train(model, split_data, split_data.src_train_x,
		split_data.src_train_y, epochs=epochs, loss=loss)


def rolling_accuracy(model, split_data, interval=1000, verbose=False):
	# Compute the accuracy of a model on various intervals of the data,
	# e.g. to check if there's domain shift.
	print("rolling accuracies")
	upper_idx = int(split_data.inter_x.shape[0] / interval)
	accs = []
	for i in range(upper_idx):
		cur_xs = split_data.inter_x[interval*i:interval*(i+1)]
		cur_ys = split_data.inter_y[interval*i:interval*(i+1)]
		_, a = model.evaluate(cur_xs, cur_ys, verbose=verbose)
		accs.append(a)
	return accs


def plot_linear(points, labels, model, index):
	# Plot the decision boundary of a linear model, and a set of data points.
	plt.cla()
	# Plot data point colored by labels.
	class_vals = [[], []]
	for i in range(2):
		class_vals[i] = [p for p, l in zip(points, labels) if l == i]
		plt.scatter([p[0] for p in class_vals[i]], [p[1] for p in class_vals[i]])
	# Plot the classifier line.
	weights = model.get_weights()[0]
	biases = model.get_weights()[1]
	x_coeff = weights[0][0] - weights[0][1]
	y_coeff = weights[1][0] - weights[1][1]
	bias = biases[0] - biases[1]
	line_func = lambda x: (-bias - x_coeff * x) / y_coeff
	xs = [p[0] for p in points]
	ys = [line_func(x) for x in xs]
	plt.plot(xs, ys)
	plt.tight_layout()
	plt.savefig('img'+str(index))


def bootstrap_once(model, pred_model, unsup_x, confidence_q=0.1, epochs=20):
	# Do one bootstrapping step on unsup_x, where pred_model is used to make predictions,
	# and we use these predictions to update model.
	logits = pred_model.predict(np.concatenate([unsup_x]))
	confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)
	alpha = np.quantile(confidence, confidence_q)
	indices = np.argwhere(confidence >= alpha)[:, 0]
	preds = np.argmax(logits, axis=1)
	model.fit(unsup_x[indices], preds[indices], epochs=epochs, verbose=False)


def bootstrap_model_iteratively(model, pred_model, split_data, interval=1000, confidence_q=0.1,
	epochs=20, loss='hinge'):
	# Iteratively bootstrap, that is bootstrap on the first `interval' samples a few times,
	# then on the next `interval' samples a few times, etc.
	print("bootstrapping model")
	unsup_x = np.concatenate([split_data.inter_x, split_data.target_unsup_x])
	debug_unsup_y = np.concatenate([split_data.inter_y, split_data.debug_target_unsup_y])
	upper_idx = int(unsup_x.shape[0] / interval)
	def entropy_regularized_ce(y_true,y_pred):
		ce = losses.sparse_categorical_crossentropy(y_true, y_pred)
		entropy = -tf.reduce_sum(y_pred * tf.log(tf.clip_by_value(y_pred, 1e-10, 1-1e-10)), axis=1)
		return ce + 0.0 * entropy
	if loss == 'hinge':
		loss = sparse_categorical_hinge(model.output_shape[1])
	else:
		loss = losses.sparse_categorical_crossentropy
	model.compile(optimizer='adam',
	              loss=[loss],
	              metrics=[metrics.sparse_categorical_accuracy])
	for i in range(upper_idx):
		cur_xs = unsup_x[interval*i:interval*(i+1)]
		cur_ys = debug_unsup_y[interval*i:interval*(i+1)]
		bootstrap_once(model, pred_model, cur_xs, confidence_q, epochs)
		model.evaluate(cur_xs, cur_ys)
		pred_model = model
		# print(np.linalg.norm(model.layers[-1].get_weights()[0]))
		# print(model.layers[-1].get_weights()[0])
		# plot(cur_xs[indices], cur_ys[indices], model, index=i)
	print('Evaluate on target.')
	model.evaluate(split_data.target_val_x, split_data.target_val_y)


def bootstrap_to_all_unsup(model, pred_model, split_data, confidence_q=0.2, epochs=20, num_inter=1000,
	loss='hinge'):
	# Bootstrap to all the unsupervised data in one go.
	print("bootstrapping to unsup data")
	if loss == 'hinge':
		loss = sparse_categorical_hinge(model.output_shape[1])
	else:
		loss = losses.sparse_categorical_crossentropy
	model.compile(optimizer='adam',
	              loss=[loss],
	              metrics=[metrics.sparse_categorical_accuracy])
	unsup_x = np.concatenate(
		[split_data.inter_x[-num_inter:], split_data.target_unsup_x],
		axis=0)
	bootstrap_once(model, pred_model, unsup_x, confidence_q, epochs)
	model.evaluate(split_data.target_val_x, split_data.target_val_y)


# Dataset-configs

Dataset = collections.namedtuple('Dataset',
	'get_data n_src_train n_src_valid n_target_unsup n_target_val n_target_test target_end '
	'n_classes input_shape')

SplitData = collections.namedtuple('SplitData',
	('src_train_x src_val_x src_train_y src_val_y target_unsup_x target_val_x final_target_test_x '
	 'debug_target_unsup_y target_val_y final_target_test_y inter_x inter_y'))

faces = Dataset(
	get_data = lambda: load_faces_data(),
	n_src_train = 1000,
	n_src_valid = 1000,
	n_target_unsup = 2000,
	n_target_val = 1000,
	n_target_test = 1000,
	target_end=18000,
	n_classes=2,
	input_shape=(32,32,1),
)

rot_mnist_0_35_29000 = Dataset(
	get_data = lambda: make_rotated_mnist(0.0, 35.0, 29000),
	n_src_train = 2000,
	n_src_valid = 1000,
	n_target_unsup = 2000,
	n_target_val = 1000,
	n_target_test = 1000,
	target_end=29000,
	n_classes=10,
	input_shape=(28,28,1),
)

gauss_2D_10K_high_noise = Dataset(
	get_data = lambda: make_moving_gaussians(
		[[2, -1], [-2, 1]], [0.8, 0.8], [[-2, -1], [2, 1]], [0.8, 0.8], 10000),
	n_src_train = 500,
	n_src_valid = 500,
	n_target_unsup = 1000,
	n_target_val = 500,
	n_target_test = 500,
	target_end=10000,
	n_classes=2,
	input_shape=(2,),
)

gauss_50D_20K_low_noise = Dataset(
	get_data = high_d_gaussians(50, 0.8, 20000),
	n_src_train = 1000,
	n_src_valid = 1000,
	n_target_unsup = 2000,
	n_target_val = 1000,
	n_target_test = 1000,
	target_end=20000,
	n_classes=2,
	input_shape=(50,),
)


# Experiment Configs.

ExperimentConfig = collections.namedtuple('ExperimentConfig',
	'dataset model interval epochs l2_reg loss')

def rotated_mnist_35_linear_experiment(seed=1):
	rand_seed(seed)
	env_config = ExperimentConfig(
		dataset=rot_mnist_0_35_29000,
		model=linear_model,
		interval=2000,
		epochs=20,
		l2_reg=0.02,
		loss='hinge')
	print("\n\n Gradual bootstrap:")
	gradual_bootstrap(env_config)
	print("\n\n Direct boostrap to target:")
	direct_bootstrap(env_config, num_inter=0, num_rounds=12)
	print("\n\n Direct boostrap to all unsup data:")
	direct_bootstrap(env_config, num_inter=22000, num_rounds=12)


def rotated_mnist_35_nneighbor_experiment(seed=1):
	rand_seed(seed)
	env_config = ExperimentConfig(
		dataset=rot_mnist_0_35_29000,
		model=nearest_neighbor.NNModel,
		interval=2000,
		epochs=20,
		l2_reg=0.02,
		loss='hinge')
	print("\n\n Gradual bootstrap:")
	gradual_bootstrap(env_config)


def rotated_mnist_35_conv_experiment(seed=1):
	rand_seed(seed)
	env_config = ExperimentConfig(
		dataset=rot_mnist_0_35_29000,
		model=simple_conv_model,
		interval=2000,
		epochs=20,
		l2_reg=0.02,
		loss='ce')
	print("\n\n Gradual bootstrap:")
	gradual_bootstrap(env_config)
	print("\n\n Direct boostrap to target:")
	direct_bootstrap(env_config, num_inter=0, num_rounds=12)
	print("\n\n Direct boostrap to all unsup data:")
	direct_bootstrap(env_config, num_inter=22000, num_rounds=12)


def gauss_2D_high_noise_linear_experiment(seed=1):
	rand_seed(seed)
	env_config = ExperimentConfig(
		dataset=gauss_2D_10K_high_noise,
		model=linear_model,
		interval=1000,
		epochs=200,
		l2_reg=0.5,
		loss='hinge')
	gradual_bootstrap(env_config)


def gauss_2D_high_noise_nn_experiment(seed=1):
	rand_seed(seed)
	env_config = ExperimentConfig(
		dataset=gauss_2D_10K_high_noise,
		model=nearest_neighbor.NNModel,
		interval=100,
		epochs=200,
		l2_reg=0.5,
		loss='hinge')
	gradual_bootstrap(env_config)


def gauss_50D_low_noise_linear_experiment(seed=1):
	rand_seed(seed)
	env_config = ExperimentConfig(
		dataset=gauss_50D_20K_low_noise,
		model=linear_model,
		interval=2000,
		epochs=200,
		l2_reg=0.5,
		loss='hinge')
	gradual_bootstrap(env_config)


def gauss_50D_low_noise_linear_experiment(seed=1):
	rand_seed(seed)
	env_config = ExperimentConfig(
		dataset=gauss_50D_20K_low_noise,
		model=nearest_neighbor.NNModel,
		interval=2000,
		epochs=200,
		l2_reg=0.5,
		loss='hinge')
	gradual_bootstrap(env_config)


def faces_linear_experiment(seed=1):
	rand_seed(seed)
	env_config = ExperimentConfig(
		dataset=faces,
		model=linear_model,
		interval=1000,
		epochs=20,
		l2_reg=0.02,
		loss='hinge')
	gradual_bootstrap(env_config)


def faces_conv_experiment(seed=1):
	rand_seed(seed)
	env_config = ExperimentConfig(
		dataset=faces,
		model=simple_conv_model,
		interval=2000,
		epochs=20,
		l2_reg=0.02,
		loss='ce')
	gradual_bootstrap(env_config)
	print("\n\n Direct boostrap to target:")
	direct_bootstrap(env_config, num_inter=0, num_rounds=7)
	print("\n\n Direct boostrap to all unsup data:")
	direct_bootstrap(env_config, num_inter=22000, num_rounds=7)


def direct_bootstrap(env_config, num_inter, num_rounds=10):
	dataset = env_config.dataset
	split_data = get_split_data(dataset)
	model = env_config.model(
		dataset.n_classes, input_shape=dataset.input_shape, l2_reg=env_config.l2_reg)
	run_model(model, split_data, epochs=env_config.epochs, loss=env_config.loss)
	pred_model = model
	for i in range(num_rounds):
		target_bootstrap = env_config.model(
			dataset.n_classes, input_shape=dataset.input_shape, l2_reg=env_config.l2_reg)
		bootstrap_to_all_unsup(
			target_bootstrap, pred_model, split_data, epochs=env_config.epochs,
			num_inter=num_inter, loss=env_config.loss)
		pred_model = target_bootstrap


def gradual_bootstrap(env_config):
	dataset = env_config.dataset
	split_data = get_split_data(dataset)
	model = env_config.model(
		dataset.n_classes, input_shape=dataset.input_shape, l2_reg=env_config.l2_reg)
	model_bootstrap = env_config.model(
		dataset.n_classes, input_shape=dataset.input_shape, l2_reg=env_config.l2_reg)
	run_model(model, split_data, epochs=env_config.epochs, loss=env_config.loss)
	bootstrap_model_iteratively(
		model_bootstrap, model, split_data,
		interval=env_config.interval, epochs=env_config.epochs, loss=env_config.loss)


def main():
	rotated_mnist_35_conv_experiment()
	# rotated_mnist_35_linear_experiment()
	# rotated_mnist_35_nneighbor_experiment()
	# gauss_2D_high_noise_linear_experiment()
	# gauss_50D_low_noise_linear_experiment()
	# faces_linear_experiment()
	# faces_conv_experiment()
	# gauss_2D_high_noise_nn_experiment()
	# gauss_50D_low_noise_linear_experiment()


if __name__ == "__main__":
	main()
