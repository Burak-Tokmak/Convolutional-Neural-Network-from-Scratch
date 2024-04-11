import numpy as np
from layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        '''
        Reshape input to the output shape
        '''
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        '''
        Reshape output to the input shape
        '''
        return np.reshape(output_gradient, self.input_shape)