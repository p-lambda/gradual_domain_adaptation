
import numpy as np
from keras.models import load_model


def self_train(model, pred_model, unsup_x, confidence_q=0.1, epochs=20):
	# Do one bootstrapping step on unsup_x, where pred_model is used to make predictions,
	# and we use these predictions to update model.
	logits = pred_model.predict(np.concatenate([unsup_x]))
	confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)
	alpha = np.quantile(confidence, confidence_q)
	indices = np.argwhere(confidence >= alpha)[:, 0]
	preds = np.argmax(logits, axis=1)
	model.fit(unsup_x[indices], preds[indices], epochs=epochs, verbose=False)


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


