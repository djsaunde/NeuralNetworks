""" This tutorial introduces the LeNet5 architecture using Theano.
LeNet5 is a convolutional neural network, good for classifying images.
This tutorial shows how to build the architecture, and comes with all
the hyperparameters you need to reproduce the paper's MNIST results.

This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling by max
 - Digit classification is implemented with a logistic regression layer rather than
 an RBF (radial basis function) network
 - LeNet5 did not have fully-connected convolutions at the second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

from __future__ import print_function

import os, sys, timeit, numpy, theano, theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

class LeNetConvPoolLayer(object):
	""" Pool layer of a convolutional network. """

	def __init__(self, random_number_generator, input, filter_shape, image_shape, poolsize = (2, 2)):
		"""
		Allocate a LeNetConvPoolLayer with shared variable internal parameters.

		:param random_number_generator: a random number generator used to allocate weights
		:param input: symbolic image tensor of shape image_shape
		:param filter_shape: (number of filters, number of input feature maps, filter height, filter width)
		:param image_shape: (batch size, number of input feature maps, image height, image width)
		:param poolsize: the downsampling / pooling factor

		"""

		assert image_shape[1] == filter_shape[1]
		self.input = input

		# there are "number of input feature maps * filter height * filter width" inputs to each hidden unit.
		fan_in = numpy.prod(filter_shape[1:])

		# each unit in the lower layer receives a gradient from: "number of output feature maps * filter height *
		# filter width / pooling size
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // numpy.prod(poolsize))

		# initialize synapses with random weights
		W_bound = numpy.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(
			numpy.asarray(
				random_number_generator.uniform(low = -W_bound, high = W_bound, size = filter_shape),
				dtype = theano.config.floatX
			),
			borrow = True
		)

		# the bias is a 1D tensor -- one bias per output feature map
		b_values = numpy.zeros((filter_shape[0],), dtype = theano.config.floatX)
		self.b = theano.shared(b_values, borrow = True)

		# convolve input feature maps with filters
		conv_out = conv2d(
			input = input,
			filters = self.W,
			filter_shape = filter_shape,
			input_shape = image_shape
		)

		# pool each feature map individually, using maxpooling
		pooled_out = pool.pool_2d(
			input = conv_out,
			ds = poolsize,
			ignore_border = True
		)

		# add the bias term. since the bias is a vector (a 1D array), we first reshape it to a tensor of shape
		# (1, number_filters, 1, 1). each bias will thus be broadcast across minibatches and feature map width
		# and height.
		self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		# store parameters of this layer
		self.params = [self.W, self.b]

		# keep track of the model's input
		self.input = input


def evaluate_lenet5(learning_rate = 0.1, number_epochs = 200, dataset = '../../data/mnist.pkl.gz', number_kernels = [20, 50],
                    batch_size = 500):
	""" Demonstrates LeNet5 on MNIST dataset.
	:param learning_rate: learning rate used (factor for the stochastic gradient descent algorithm)
	:param number_epochs: maximal number of epochs to run the optimizer
	:param data_set: path to the dataset to use for training / testing
	:param number_kernels: number of kernels on each layer

	"""

	random_number_generator = numpy.random.RandomState(23455)

	datasets = load_data(dataset)

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]
	n_train_batches //= batch_size
	n_valid_batches //= batch_size
	n_test_batches //= batch_size

	# allocate symbolic variables for the data
	index = T.lscalar()

	x = T.matrix('x') # the data is presented as rasterized images
	y = T.ivector('y') # the labels are presented as a 1D vector of integer labels

	# building the model
	print('...building the model')

	# reshape matrix of rasterized images of shape (batch_size, 28 * 28) to a 4D tensor, compatible with our
	# LeNetConvPoolLayer. (28, 28) is the size of our MNIST images.
	layer0_input = x.reshape((batch_size, 1, 28, 28))

	# construct the first convolutional pooling layer: filtering reduces the image size to (28-5+1, 28-5+1) = (24, 24),
	# and maxpooling further reduces this to (24/2, 24/2) = (12, 12). The 4D output tensor is thus of shape
	# (batch_size, number_kernels[0], 12, 12).
	layer0 = LeNetConvPoolLayer(
		random_number_generator,
		input = layer0_input,
		image_shape = (batch_size, 1, 28, 28),
		filter_shape = (number_kernels[0], 1, 5, 5),
		poolsize = (2, 2)
	)

	# Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
	layer1 = LeNetConvPoolLayer(
        random_number_generator,
        input = layer0.output,
        image_shape=(batch_size, number_kernels[0], 12, 12),
        filter_shape=(number_kernels[1], number_kernels[0], 5, 5),
        poolsize=(2, 2)
    )

	# the hidden layer will be fully connected, and so it operates on 2D matrices of shape (batch_size, number of
	# pixels) (i.e., matrix of rasterized images). This will generate a matrix of shape (batch_size, number_kernels[1]
	# * 4 * 4), or (500, 50 * 4 * 4) = (500, 800) with the default values.
	layer2_input = layer1.output.flatten(2)

	# construct a fully connected sigmoidal layer
	layer2 = HiddenLayer(
		random_number_generator,
		input = layer2_input,
		number_ins = number_kernels[1] * 4 * 4,
		number_outs = 500,
		activation = T.tanh
	)

	# classify the values of the fully connected sigmoidal layer
	layer3 = LogisticRegression(input = layer2.output, number_ins = 500, number_outs = 10)

	# the cost we minimize during training is the negative log-likelihood of the model
	cost = layer3.negative_log_likelihood(y)

	# create a function to compute the mistakes that are made by the model
	test_model = theano.function(
		[index],
		layer3.errors(y),
		givens={
			x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	validate_model = theano.function(
		[index],
		layer3.errors(y),
		givens={
			x: valid_set_x[index * batch_size: (index + 1) * batch_size],
			y: valid_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	# create a list of all model parameters to be fit by gradient descent
	params = layer3.params + layer2.params + layer1.params + layer0.params

	# create a list of gradients for all model parameters
	grads = T.grad(cost, params)

	# train_model is a function that updates the model parameters by
	# SGD. Since this model has many parameters, it would be tedious to
	# manually create an update rule for each model parameter. We thus
	# create the updates list by automatically looping over all
	# (params[i], grads[i]) pairs.
	updates = [
		(param_i, param_i - learning_rate * grad_i)
		for param_i, grad_i in zip(params, grads)
	]

	train_model = theano.function(
		[index],
		cost,
		updates=updates,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	# train the model
	print('...training the model')

	# early stopping parameters
	patience = 10000 # look at this many exemplars regardless
	patience_increase = 2 # wait this long when a new best model is found
	improvement_threshold = 0.995 # an improvement of this magnitude is considered significant
	validation_frequency = min(n_train_batches, patience // 2)

	best_validation_loss = numpy.inf
	best_iter = 0
	test_score = 0.
	start_time = timeit.default_timer()

	epoch = 0
	done_looping = False

	while epoch < number_epochs and not done_looping:
		epoch = epoch + 1
		for minibatch_index in range(n_train_batches):
			iter = (epoch - 1) * n_train_batches + minibatch_index

			if iter % 100 == 0:
				print('training @ iter = ', iter)
			cost_ij = train_model(minibatch_index)

			if (iter + 1) % validation_frequency == 0:
				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i in range(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_losses)
				print('epoch %i, minibatch %i/%i, validation error %f %%' %
						(epoch, minibatch_index + 1, n_train_batches,
						this_validation_loss * 100.))

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:
					# improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss * improvement_threshold:
						patience = max(patience, iter * patience_increase)

					# save best validation score and iteration number
					best_validation_loss = this_validation_loss
					best_iter = iter

					# test it on the test set
					test_losses = [
						test_model(i) for i in range(n_test_batches)
					]
					test_score = numpy.mean(test_losses)
					print(('     epoch %i, minibatch %i/%i, test error of '
							'best model %f %%') %
							(epoch, minibatch_index + 1, n_train_batches,
							test_score * 100.))

			if patience <= iter:
				done_looping = True
				break

	end_time = timeit.default_timer()
	print('Optimization complete.')
	print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

if __name__ == '__main__':
	evaluate_lenet5()


def experiment(state, channel):
	evaluate_lenet5(state.learning_rate, dataset=state.dataset)