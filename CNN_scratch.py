import numpy as np
from layer import Layer
from scipy import signal

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth): # depth is a number representing how many kernels that we want (and therefore the depth of our output)
        input_depth, input_height, input_width = input_shape # unpacking the input shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1) # ex: --> 3x3(input) * 2x2(kernel) ==> 2x2(output) [stride=1]
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size) # number of kernels, depth of each kernel (equal to the depth of the input), size of the matrices contained inside each kernel
        self.kernels = np.random.randn(*self.kernels_shape) # initialize kernels randomly
        self.biases = np.random.randn(*self.output_shape) # initialize biases randomly (biases have the same shape as the output)
    
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases) # start by copying the current biases since each output is equal to the bias + "something_else"
        for i in range(self.depth):
            for j in range(self.input_depth):
                '''
                The convolution between the input matrix I and kernel matrix K is the cross correlation between I and 180 degree rotated version of K.
                "Valid" cross correlation is used here. In valid cross correlation you start computing the product by placing the kernel entrely
                on top of the input, and stop sliding the kernel when it hits the border of the input.
                '''
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid") # "something_else"
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # TODO: Update the trainable parameters (kernels & biases) and return input gradient
        #1) Initializing empty arrays for the kernel gradient and the input geradient
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        #2) Given the error of the network(E) we need to calculate two things:
        #    - Derivative of E with respect to the trainable parameters of the layer
        #    - Derivative of E with respect to the input of the layer
        # this way the previous layer can take it as the derivative of E with respect to its own output
        # and perform the same operations
        '''
        Compute the derivative of the error of the network with respect to Kij (each element of the kernel(s))
        (No need to calculate the bias gradient since it is equal to the output gradient)
        **************
        * k11 * k12  *
        *     *      *
        **************
        * k21 * k22  *
        *     *      *
        **************
        '''
        for i in range(self.depth):
            for j in range(self.input):
                kernels_gradient[i, j] =signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        #3 Update the kernels and biases using gradient descent, then return the input gradient
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient