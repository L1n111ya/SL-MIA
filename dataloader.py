from cProfile import label
import tensorflow as tf


def load_MNIST(data_dir):
    """
    Load MNIST dataset and data preprocessing.
    :param data_dir: Path where to find the data file.
    :return: Tuple of numpy arrays: (x_train, y_train), (x_test, y_test).
    """
    # Initialize Data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(data_dir)
    
    x_train = tf.image.resize(x_train, (64, 64))
    y_train = tf.keras.utils.to_categorical(y_train, num_class=8)
    
    x_test = tf.image.resize(x_test, (64, 64))
    y_test = tf.keras.utils.to_categorical(y_test, num_class=8)
    
    return (x_test, y_train), (x_test, y_test)


def load_CIFAR10(data_dir):
    """
    Load CIFAR10 dataset and data preprocessing.
    :param data_dir: Path where to find the data file.
    :return: Tuple of numpy arrays: (x_train, y_train), (x_test, y_test).
    """
    # Initialize Data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data(data_dir)
    
    y_train = tf.keras.utils.to_categorical(y_train, num_class=10)
    y_test = tf.keras.utils.tp_categorical(y_test, num_class=10)
    
    return (x_train, y_train), (x_test, y_test)


def load_CIFAR100(data_dir):
    """
    Load CIFAR100 dataset and data preprocessing.
    :param data_dir: Path where to find the data file.
    :return: Tuple of numpy arrays: (x_train, y_train), (x_test, y_test).
    """
    # Initialize Data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(data_dir, label_mode='fine')
    
    y_train = tf.keras.utils.to_categorical(y_train, num_class=100)
    y_test = tf.keras.utils.to_categorical(y_test, num_class=100)
    
    return (x_train, y_train), (x_test, y_test)
