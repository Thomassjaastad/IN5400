#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 1 in                                             #
# IN5400 - Machine Learning for Image analysis                                  #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2019.02.12                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Implementation of convolution forward and backward pass"""

import numpy as np

def conv_layer_forward(input_layer, weight, bias, pad_size=1, stride=1):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of M data points, each with C channels, height H and
    width W. We convolve each input with C_o different filters, where each filter
    spans all C_i channels and has height H_w and width W_w.

    Args:
        input_alyer: The input layer with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
    """
    # TODO: Task 2.1

    #                C_x        H_x      W_x
    (batch_size, channels_x, height_x, width_x) = input_layer.shape
    
    #    N = C_y             H_w = 2K+1 W_w = 2K+1      
    (num_filters, channels_w, height_w, width_w) = weight.shape

        
    K = int((weight.shape[0] - 1)/2)
    x = np.pad(input_layer, [(0, 0), (0, 0), (1, 1), (1, 1)], 'constant', constant_values = 0)
        
    width_y = int(1 + (width_x + 2*pad_size - width_w)/stride) 
    height_y = int(1 + (height_x + 2*pad_size - height_w)/stride)
    output_layer = np.zeros((batch_size, num_filters, height_y, width_y))    # Should have shape (batch_size, num_filters, height_y, width_y)
    
    for b in range(batch_size):
        for n in range(num_filters):
            output_layer[b,n] = bias[n]
            for c in range(channels_x):
                for wy in range(width_y):
                    for hy in range(height_y):
                        temp = 0
                        for hw in range(width_w):
                            for ww in range(height_w):
                                temp += x[b, c, hy*stride + hw, wy*stride + ww]*weight[n, c, hw, ww]
                        output_layer[b, n, hy, wy] += temp
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")

    return output_layer


def conv_layer_backward(output_layer_gradient, input_layer, weight, bias, pad_size=1):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Args:
        output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
            (batch_size, num_filters, height_y, width_y)
        input_layer: Input layer x with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        input_layer_gradient: Gradient of the loss L with respect to the input layer x
        weight_gradient: Gradient of the loss L with respect to the filters w
        bias_gradient: Gradient of the loss L with respect to the biases b
    """
    # TODO: Task 2.2
    input_layer_gradient, weight_gradient, bias_gradient = None, None, None

    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    batch_size, channels_x, height_x, width_x = input_layer.shape
    num_filters, channels_w, height_w, width_w = weight.shape

    assert num_filters == channels_y, (
        "The number of filters must be the same as the number of output layer channels")
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")
    bias_gradient = np.sum(output_layer_gradient, axis =(0, 2, 3)) 
    weight_gradient = np.zeros_like(weight)
    input_layer_gradient = np.zeros_like(input_layer)
    x_padded = np.pad(input_layer, [(0, 0), (0, 0), (1, 1), (1, 1)], 'constant', constant_values = 0)

    # computing the weight gradients 
    for b in range(batch_size):
        for n in range(num_filters):
            for c in range(channels_x):
                for x in range(width_w): 
                    for y in range(height_w):
                        temp_grad_W = 0
                        for p in range(width_x):
                            for q in range(height_x):
                                temp_grad_W += np.sum(output_layer_gradient[b, n, p, q]*x_padded[b, c, p + x, q + y])
                        weight_gradient[n, c, x, y] += temp_grad_W
    
    # flipping weights. Makes it possible to convolve.
    weight_flipped = np.flip(weight, axis=2)
    weight_flipped = np.flip(weight_flipped, axis=3)
    out_padded = np.pad(output_layer_gradient, [(0, 0), (0, 0), (1,1), (1,1)], 'constant', constant_values = 0)
    
    # computing inout gradients 
    for b in range(batch_size):
        for c in range(channels_x):
            for p in range(width_x):
                for q in range(height_x): 
                    temp_grad_X = 0
                    for n in range(num_filters):  
                        for x in range(width_w): 
                            for y in range(height_w):
                                temp_grad_X += np.sum(out_padded[b, n, p+x, q+y]*weight_flipped[n, c, x, y])
                    input_layer_gradient[b, c, p, q] += temp_grad_X

    return input_layer_gradient, weight_gradient, bias_gradient


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
