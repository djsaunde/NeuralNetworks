"""
Dan Saunders
5 July 2016

Building a multi-layer perceptron implementation from the tutorial found at http://deeplearning.net/tutorial/mlp.html
Most code will be duplicate. Re-implementing to solidify understanding.
"""

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
        