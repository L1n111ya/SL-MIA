from tabnanny import verbose
from turtle import shape
import numpy as np
from tqdm import tqdm 
import tensorflow as tf
from tensorflow import keras
from six.moves import cPickle

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='1' 

def cifar10_load_data(data_dir='./data', idx=1):
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    def cifar10_load_batch(batch_file, label_key='labels'):
        """Internal utility for parsing CIFAR data.
        # Arguments
            batch_file: path to the batch file to parse.
            label_key: key for label data in the retrieve dictionary.
        # Returns
            A tuple `(data, labels)`.
        """
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

    num_train_samples = 10000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    batch_file = './data/data_batch_' + str(idx)
    x_train, y_train = cifar10_load_batch(batch_file)

    batch_file = './data/test_batch'
    x_test, y_test = cifar10_load_batch(batch_file)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

def train_model(x_train, y_train, x_test, y_test, idx):
    if idx == 1:
        model_name = './model/cifar10_trained_model' + str(5) + '.h5'
    else:
        model_name = './model/cifar10_trained_model' + str(idx-1) + '.h5'
    save_model = './model/cifar10_trained_model' + str(idx) + '.h5'

    # The data, split between train and test sets:
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Initialization
    batch_size = 32
    num_classes = 10  
    epochs = 10

    # Models
    # if idx == 1:
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(32, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Activation('softmax'))
    # else:
    model = keras.models.load_model(model_name)

    # Initiate RMSprop optimizer
    opt = keras.optimizers.SGD(learning_rate=0.1)

    # Let's train the model using RMSprop
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          verbose=1)

    model.save(save_model)
    print('Saved trained model at %s ' % save_model)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
