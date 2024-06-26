import random
import numpy as np
from sympy import beta, re
import tensorflow as tf
import cv2 as cv
from functools import partial


def gaussian_noisy(inputs_set, mean=0, var=0.01):
    """Get the gussian noisy for the inputs_set.
    Args:
        inputs_set (enumerate): The input dataset.
        mean (int): The mean value of the gussian noise. Defaults to 0.
        var (float): The varicance of the gussian noise. Defaults to 0.01.
    Returns:
        Tuple of numpy array: The inputs_set which added to the gussian noise.
    """
    ret = np.empty(inputs_set.shape)
    for m, input in enumerate(inputs_set):
        noise = np.random.normal(mean, var**0.5, input.shape)
        out = input + noise
        if out.min() < 0:
            low_clip = -1
        else:
            low_clip = 0
        out = np.clip(out, low_clip, 1.0)
        ret[m, :] = out
        
    return ret


def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples].
    """
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    # Computing the pair-wise distances, we should ues the broadcasting.    
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    """Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
      A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    # Using a sigmas list to have a better result.
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    
    dist = compute_pairwise_distances(x, y)
    
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    '''Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.
    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
    where K = <\phi(x), \phi(y)>,
      is the desired kernel function, in this case a radial basis kernel.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
                GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    '''
    