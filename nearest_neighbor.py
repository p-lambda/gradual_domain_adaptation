
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class NNModel:

	def __init__(self, num_classes, input_shape, l2_reg, k=5):
		# self._num_classes = num_classes
		self._classifier = KNeighborsClassifier(n_neighbors=k)
		self.output_shape = [None, 2]

	def _flatten(self, x):
		s = x.shape
		x = np.reshape(x, (s[0], np.prod(s[1:])))
		return x

	def compile(self, optimizer, loss, metrics):
		pass

	def fit(self, train_x, train_y, epochs, verbose=True):
		train_x = self._flatten(train_x)
		self._classifier.fit(train_x, train_y)

	def evaluate(self, test_x, test_y, verbose=True):
		test_x = self._flatten(test_x)
		acc = self._classifier.score(test_x, test_y)
		print("Accuracy was: %.3f" % acc)
		return None, acc

	def predict(self, x):
		x = self._flatten(x)
		return self._classifier.predict_proba(x)
