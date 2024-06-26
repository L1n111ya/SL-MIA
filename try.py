import sys
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='3' 
import tensorflow as tf
from tensorflow import keras
from six.moves import cPickle
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

def cifar10_load_data(data_dir='./data', idx=1):
    """Loads CIFAR10 dataset"""

    def cifar10_load_batch(batch_file, label_key='labels'):
        """Dncode CIFAR10 dataset"""

        with open(batch_file, 'rb') as f:
            if sys.version_info < (3,):
                d = cPickle.load(f)
            else:
                d = cPickle.load(f, encoding='bytes')
                # decode utf8
                d_decoded = {}
                for k, v in d.items():
                    d_decoded[k.decode('utf8')] = v
                d = d_decoded

        data = d['data']
        labels = d[label_key]

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        batch_file = './cifar-10-batches-py/data_batch_' + str(idx)
        (
            x_train[(i - 1) * 10000: i * 10000, :, :, :]
          , y_train[(i - 1) * 10000: i * 10000]
        ) = cifar10_load_batch(batch_file)

    batch_file = './cifar-10-batches-py/test_batch'
    x_test, y_test = cifar10_load_batch(batch_file)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def train_model(x_train, y_train, x_test, y_test, idx, x):
    model_name = './model/trained_model' + str(idx-1) + '.h5'
    save_model = './model/trained_model' + str(idx) + '.h5'

    # The data, split between train and test sets:
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Initialization
    batch_size = 64
    num_classes = 10  
    epochs = 10

    # Models
    model = keras.Sequential([
        keras.applications.DenseNet121(include_top=False,
            weights=None,
            input_shape=x_train[1].shape),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(num_classes),
            keras.layers.Activation('sigmoid')
            ])
    #model.summary()

    if idx > 1:
        model = keras.models.load_model(model_name)

    # Initiate RMS optimizer
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_split=0.2,
            verbose=1)

    model.save(save_model)
    print('Saved trained model at %s ' % save_model)

    # Score trained model.
    #scores = model.evaluate(x_test, y_test, verbose=1)
    #print('Test loss:', scores[0])
    #print('Test accuracy:', scores[1])
    out = model.predict(x.reshape(1, 32, 32, 3))
    print('The data predict:', out)


from cv2 import Subdiv2D_PREV_AROUND_DST
import tensorflow as tf
from tensorflow import keras
import numpy as np
from functools import partial


def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
      ValueError: if the inputs do no matched the specified dimensions.
    """
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)

    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
      A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)

    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    '''
    Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
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
    with tf.name_scope('MaximumMeanDiscrepancy'):
        # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))

        # We do not allow the loss to become negative.
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def mmd_loss(source_samples, target_samples):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

    loss_value = maximum_mean_discrepancy(
        source_samples, target_samples, kernel=gaussian_kernel)

    return loss_value


def MIA(pre_model, sub_model, test_data, target_data):
    """
    The major membership inference attack for swarm learning with MMD.

    """
    x_test, y_test = test_data
    x_target, y_target = target_data
    dValue_1st, dValue_2nd = np.zeros(len(x_target)), np.zeros(len(x_target))
    pred = np.zeros(len(x_target))
    accuracy = keras.metrics.Accuracy()
    precision = keras.metrics.Precision()
    recall = keras.metrics.Recall()
    
    for i in range(len(x_test)):
        pre = pre_model.predict(x_target)
        sub = sub_model.predict(x_target)
        for k in range(len(x_target)):
            for j in range(10):
                if pre[k, j] == j:
                    pre_pred = j
                if sub[k, j] == j:
                    sub_pred = j
        
        pre_model.compile(loss='sparse_categorical_crossentropy',
                optimizer=keras.optimizers.SGD(learning_rate=0.1),
                metrics=['accuracy'])
        x, y = x_target[i], y_target[i]
        x, y = x[np.newaxis, :], y[np.newaxis, :]
        pre_model.fit(x, y)

        new_pred = pre_model.predict(x_test)
        dValue_1st[i] = mmd_loss(pre_pred, sub_pred)
        dValue_2nd[i] = mmd_loss(new_pred, sub_pred)
        if dValue_1st[i] > dValue_2nd[i]:
            pred[i] = 0
        else:
            pred[i] = 1
        
        accuracy.update_state(y_target, pred)
        precision.update_state(y_target, pred)
        recall.update_state(y_target, pred)
        F1_Score = 2 * (precision.result() * recall.result()) / (precision.result() + recall.result())
        print('accuracy:%.4f precision:%.4f recall:%.4f F1_Score:%.4f'
                % (accuracy.result(), precision.result(), recall.result(), F1_Score))


import random
import numpy as np
from tensorflow import keras
import os
os.environ['CUDA_VISIBLE_DEVICES']='1' 


batch_file = './data'
(x, y), (x_test, y_test) = cifar10_load_data(batch_file)


# Train the target model
for i in range(1, 50):
    save_path_1 = './train/train_data_' + str(i) + '.npz'
    save_path_2 = './lie/lie_data_' + str(i) + '.npz'
    num = random.randint(0, 4)
    if num==4:
        num_lie = 0
    else:
        num_lie = num + 1
    num = num * 10000
    num_lie = num_lie * 10000
    x_train, y_train = x[num:num+10000,], y[num:num+10000,]
    x_lie, y_lie = x[num_lie:num_lie+10000], y[num_lie:num_lie+10000]
    train_model(x_train, y_train, x_test, y_test, idx=i, x=x[666])
    np.savez(save_path_1, x=x_train, y=y_train)
    np.savez(save_path_2, x=x_lie, y=y_lie)