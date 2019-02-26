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

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # TODO: Task 1.1
    params = {}
    dims = conf.get('layer_dimensions')
    N = dims[0]
    sigma = np.sqrt(2/N)
    for l in range(1, len(dims)):
        weights = np.random.normal(0, sigma, (dims[l-1], dims[l]))    
        bias = np.zeros((dims[l]))
        params["W_%i" % l] = weights
        #if (l+1) > (len(dims)-1):
        #    break
        params["b_%i" % l] = bias
    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 a)
    if activation_function == 'relu':
        return Z*(Z>=0)
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        print(activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 b)
    x = Z - np.max(Z, axis = 0)
    x_exp = np.exp(x)
    x_exp_tot = np.sum(x_exp[:, np.newaxis], axis = 0)
    t_k = x - np.log(x_exp_tot)
    s = np.exp(t_k)
    return s


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # TODO: Task 1.2 c)
    dimensions = conf.get('layer_dimensions')
    features = {}
    A = X_batch
    for l in range(1, (len(dimensions))):
        w = params["W_%i" % l]
        b = params['b_%i' % l]
        b = b[:,np.newaxis]
        Z = np.dot(w.T, A) + b
        if l < (len(dimensions) - 1):
            A = activation(Z, 'relu')
        else:
            A = softmax(Z)
        features["A_%i" % l] = A
        features["Z_%i" % l] = Z
    Y_proposed = A
    return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # TODO: Task 1.3
    cost = -1/Y_proposed.shape[1]*np.sum(np.sum(Y_reference*np.log(Y_proposed), axis=0))
    num_correct = 0
    for i in range(Y_proposed.shape[1]):
        if np.argmax(Y_proposed[:, i]) == np.argmax(Y_reference[:, i]):
            num_correct += 1
    return cost, num_correct

def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.4 a)
    if activation_function == 'relu_der':
        return 1*(Z>=0)
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # TODO: Task 1.4 b)
    grad_params = {}
    scale = 1/Y_proposed.shape[1]
    loss_lastlayer_b = np.sum((Y_proposed - Y_reference), axis = 1, keepdims = True)*scale
    loss_lastlayer_W = Y_proposed - Y_reference
    dimensions = conf.get('layer_dimensions')
    L = len(dimensions) - 1
    dl_dZ = {}
    # activation has 0, 1, 2 activations. z, b and w have 1, 2
    for l in reversed(range(1, len(dimensions))):
        w = params['W_%i' % l]
        b = params['b_%i' % l]
        A_prev = features['A_%i' % (l-1)]   #is l-1 when testing
        Z = features['Z_%i' % l]
        if l == L:
            #last layer l = 2
            grad_params['grad_b_%i' % l] = loss_lastlayer_b
            grad_params['grad_W_%i' % l] = np.dot(A_prev, loss_lastlayer_W.T)*scale 
        else:
            w_last = params['W_%i' % (l + 1)]
            temp = grad_params['grad_W_%i' % (l + 1)]
            dl_dZ['dZ_%i' % l] = np.dot(w_last, loss_lastlayer_W)
            #hidden layers l = 1
            J_zl = np.dot(dl_dZ['dZ_%i' % l], activation_derivative(Z, 'relu_der').T)*scale
            J_zl_b = np.diagonal(J_zl)
            J_zl_W = activation_derivative(Z, 'relu_der')*dl_dZ['dZ_%i' % l]
            print(J_zl_W.shape)
            grad_params['grad_b_%i' % l] = J_zl_b[:, np.newaxis]
            grad_params['grad_W_%i' % l] = scale*np.dot(A_prev, J_zl_W.T)
            loss_lastlayer_W = J_zl_W  
    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        params: Updated parameter dictionary.
    """
    # TODO: Task 1.5
    updated_params = {}
    lamb = conf['learning_rate']
    numb_layers = int(len(params)/2)
    for l in range(1, numb_layers):
        W = params['W_%i' % l]
        b = params['b_%i' % l]
        grad_w = grad_params['grad_W_%i' % l]
        grad_b = grad_params['grad_b_%i' % l]
        print(W.shape, grad_w.shape)
        #updated_params['W_%i' % l] = W -lamb*grad_w 
        #updated_params['b_%i' % l] = b -lamb*grad_b
    return updated_params
