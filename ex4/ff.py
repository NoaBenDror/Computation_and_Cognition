import numpy as np

class FF(object):
	"""A simple FeedForward neural network"""
	def __init__(self, layerDims):
		super(FF, self).__init__()
		n_weights = len(layerDims)-1
		self.weights = []
		for i in range(n_weights):
			self.weights.append(0.1*np.random.randn(layerDims[i+1], layerDims[i]))

	def sgd(self, X, y, epochs, eta, mb_size, Xtest, ytest):
		N = X.shape[1]
		n_mbs = int(np.ceil(N/mb_size))
		acc = self.eval_test(Xtest, ytest)

		updates = 0
		steps = [updates]
		test_acc = [acc]
		print("Starting training, test accuracy: {0}".format(acc))

		for i in range(epochs):
			perm = np.random.permutation(N);
			for j in range(n_mbs):
				X_mb = X[:,perm[j*mb_size:(j+1)*mb_size]]
				y_mb = y[:,perm[j*mb_size:(j+1)*mb_size]]

				grads = self.backprop(X_mb, y_mb)

				for k,grad in enumerate(grads):
					self.weights[k] = self.weights[k] - (eta/mb_size)*grad

				updates = updates + 1
				if updates%50 == 0:
					steps.append(updates)
					test_acc.append(self.eval_test(Xtest, ytest))

			acc = self.eval_test(Xtest, ytest)
			print("Done epoch {0}, test accuracy: {1}".format(i+1, acc))

		steps = np.asarray(steps)
		steps = steps/n_mbs

		return steps, test_acc


	def backprop(self,X,y):
		# X is a matrix of size input_dim*mb_size
		# y is a matrix of size output_dim*mb_size
		# you should return a list 'grads' of length(weights) such
		# that grads[i] is a matrix containing the gradients of the
		# loss with respect to weights[i].

		n_weights = len(self.weights)

		# ForwardPass
		v = X
		h_arr = []  # array of the layers before activation
		s_arr = []  # array of the layers after activation
		for weight in self.weights:
			v = np.matmul(weight, v)
			h_arr.append(v)
			v = self.activation(v)  # activation function on layer
			s_arr.append(v)

		# BackwardPass
		del_arr = []
		del_arr.append(self.loss_deriv(v, y) * self.activation_deriv(h_arr[n_weights-1]))  # output layer
		for i in range(n_weights - 2, -1, -1):
			del_arr.append(np.matmul(np.transpose(self.weights[i+1]), del_arr[-1]) * self.activation_deriv(h_arr[i]))
		del_arr.reverse()

		# Gradients
		gradients = []
		gradients.append(np.matmul(del_arr[0], np.transpose(X)))
		for i in range(1, n_weights):
			gradients.append(np.matmul(del_arr[i], np.transpose(s_arr[i-1])))

		return gradients

	def predict(self,x):
		a = x
		for w in self.weights:
			a = FF.activation(np.dot(w,a))

		return a

	def eval_test(self,Xtest, ytest):
		ypred = self.predict(Xtest)
		ypred = ypred==np.max(ypred,axis=0)
		
		return np.mean(np.all(ypred==ytest,axis=0))

	@staticmethod
	def activation(x):
		return np.tanh(x)

	@staticmethod
	def activation_deriv(x):
		return 1-(np.tanh(x)**2)

	@staticmethod
	def loss_deriv(output, target):
		# Derivative of loss function with respect to the activations
		# in the output layer.
		# we use quadratic loss, where L=0.5*||output-target||^2
		return output-target
