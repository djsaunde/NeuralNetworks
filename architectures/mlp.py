"""
Dan Saunders
5 July 2016

Building a multi-layer perceptron implementation from the tutorial found at http://deeplearning.net/tutorial/mlp.html
Most code will be duplicate. Re-implementing to solidify understanding.

This tutorial introduces the multi-layer perceptron using Theano.

A multi-layer perceptron is a logistic regressor where instead of feeding the input to the logistic regression, you insert
an intermediate layer, called the hidden layer, which has a nonlinear activation function (usually tanh(x) or sigmoid(x)). 
One can use many such hidden layers, making the architecture deep. This tutorial will tackle the problem of MNIST digit 
classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),
    
References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5
"""

from __future__ import print_function


__docformat__ = 'restructedtext en'


import os, sys, timeit, numpy, theano, theano.tensor as T


from logistic_sgd import LogisticRegression, load_data


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, random_number_generator, input, number_ins, number_outs, W=None, b=None, activation=T.tanh):
        """
        Typical hidden layer of a multi-layer perceptron: units are fully connected and have a sigmoidal 
        activation function. The weight matrix W has the shape (number_ins, number_outs), and the bias vector has
        the shape (number_outs, ).
        
        The nonlinearity used in this implementation is the tanh(x) function.
        
        Hidden unit activation is given by: tanh(dot_product(input, W) + b)
        
        :type random_number_generator: numpy.random.RandomState
        :param random_number_generator: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (number_examples, number_ins)

        :type number_ins: int
        :param number_ins: dimensionality of input

        :type number_outs: int
        :param number_outs: number of hidden units

        :type activation: theano.Op or function
        :param activation: nonlinearity to be applied in the hidden layer
        """
        
        self.input = input
        # end-snippet-1
        
        # `W` is initialized with `W_values` which is randomly sample from sqrt(-6./(n_in+n_hidden)) and 
        # sqrt(6./(n_in+n_hidden)) for the tanh(x) activation function. The output of uniform is converted 
        # using asarray to dtype theano.config.floatX so that the code is runable on GPU. Note : optimal 
        # initialization of weights is dependent on the activation function used (among other things).
        # For example, results presented in [Xavier10] suggest that you should use 4 times larger initial
        # weights for sigmoid(x) compared to tanh(x). We have no inforamtion for other functions, so we
        # use the same as tanh(x).
        
        if W is None:
            W_values = numpy.asarray(
                random_number_generator.uniform(
                    low=-numpy.sqrt(6. / (number_ins + number_outs)),
                    high=numpy.sqrt(6. / (number_ins + number_outs)),
                    size=(number_ins, number_outs)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
                
            W = theano.shared(value=W_values, name='W', borrow=True)
            
        if b is None:
            b_values = numpy.zeros((number_outs,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            
        self.W = W
        self.b = b
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        # parameters of the model
        self.params = [self.W, self.b]
        
# start-snippet-2
class MLP(object):
    """
    Multi-layer Perceptron Class
    
    A multi-layer perceptron is a feedforward artificial neural network model than has one layer or more of hidden
    units and nonlinear activations. Intermediate layers usually have as activation function the tanh(x) or the 
    sigmoid(x) function (defined here by a ``HiddenLayer`` class) while the top layer is a softmax layer (defined 
    here by a ``LogisticRegression`` class).
    """
    
    def __init__(self, random_number_generator, input, number_ins, number_hiddens, number_outs):
        """
        Initialize the parameters for the multi-layer perceptron.
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type numer_ins: int
        :param numer_ins: number of input units, the dimension of the space in
        which the datapoints lie

        :type number_hiddens: int
        :param number_hiddens: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        
        # Since we are dealing with a one hidden layer multi-layer perceptron, this will translate into a 
        # HiddenLayer with a tanh(x) activation function connected to the LogisticRegression layer; the activation
        # function can be replaced by sigmoid or any other nonlinear function.
        self.hiddenLayer = HiddenLayer(
            random_number_generator = random_number_generator,
            input = input,
            number_ins = number_ins,
            number_outs = number_outs,
            activation = T.tanh
        )
        
        # The logistic regression layer gets as input the activations of the units of the hidden layers
        self.logRegressionLayer = LogisticRegression(
            input = self.hiddenLayer.output,
            number_ins = number_ins,
            number_outs = number_outs
        )
        
        # end-snippet-2 start-snippet-3
        # L1 norm; one regularization option is to force the L1 norm to be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
        )
        
        # square of L2 norm; one regularization option is to for the L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()
        )
        
        # negative log-likelihood of the multi-layer perceptron is given by the negative log-likelihood of the 
        # output of the model, computed in the logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        
        # the parameters of the model are the parameters of the two layers it is composed of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3
        
        # keep track of the model's input
        self.input = input
        

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, number_epochs=1000, data_set='../../data/mnist.pkl.gz', 
             batch_size=20, number_hiddens=500):
    """
    Demonstrate stochastic gradient descent optimization for a multi-layer perceptron.
    
    This is demonstrated on the MNIST dataset.
    
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


    """
    
    datasets = load_data(data_set)
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # compute the number of minibatches for training, validation, and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    # build model
    print('... building the model')
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as a one dimensional vector of integer labels
    
    random_number_generator = numpy.random.RandomState(1234)
    
    # construct the MLP class
    classifier = MLP(
        random_number_generator=random_number_generator,
        input=x,
        number_ins=28 * 28,
        number_hiddens=number_hiddens,
        number_outs=10
    )
    
    # start-snippet-4
    # the cost we minimize during training is the negative log-likelihood of the model plus the 
    # regularization terms (L1 and L2); cost is expressed here symbolically
    cost = (
        classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4
    
    # compiling a Theano function that computes the mistakes that are made by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    # start-snippet-5
    # Compute the gradient of the cost with respect to theta (stored in params). The resulting gradients will
    # be stored in the list gparams.
    gparams = [T.grad(cost, param) for param in classifier.params]
    
    # specify how to update the parameters of the model with a list of (variable, update expression) pairs
    
    # given two lists of the same length, A and B, zip generates a list of the same size, where each element is
    # a pair formed from the two lists, in order
    updates = [
        (param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)
    ]
    
    # compiling a Theano function `train_model` that returns the cost, but at the same time updates the parameters
    # of the model based on rules defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    # end-snippet-5
    
    # train model
    print('... training')
    
    # early-stopping parameters
    patience = 10000  # look at this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience // 2) # go through this many minibatches before checking the
                                                               # network on the validation set (check every epoch here)
        
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()
    
    epoch = 0
    done_looping = False
    
    while epoch < number_epochs and not done_looping:
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on the validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )
                
                # if we got the best validation score up until this point
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    
                    # test it on the test set
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    
            if patience <= iter:
                done_looping = True
                break
                
    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    
if __name__ == '__main__':
    test_mlp()
                    
    
    
    
            
            
        
        
        
        
        
        
        
        
        