import os
from tabnanny import verbose
os.environ['CUDA_VISIBLE_DEVICES']='2' 
import sys
import numpy as np
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
        batch_file = './data/data_batch_' + str(idx)
        (
            x_train[(i - 1) * 10000: i * 10000, :, :, :]
          , y_train[(i - 1) * 10000: i * 10000]
        ) = cifar10_load_batch(batch_file)

    batch_file = './data/test_batch'
    x_test, y_test = cifar10_load_batch(batch_file)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def train_model(x_train, y_train, x_test, y_test, idx):
    model_name = './model/trained_model' + str(idx-1) + '.h5'
    save_model = './model/trained_model' + str(idx) + '.h5'

    # The data, split between train and test sets:
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Initialization
    batch_size = 64
    num_classes = 10  
    epochs = 10

    # Models
    model = keras.Sequential([
        keras.applications.DenseNet121(include_top=False,
            weights='imagenet',
            input_shape=x_train[1].shape),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(num_classes),
            keras.layers.Activation('softmax')
            ])
    model.summary()

    if idx > 1:
        model = keras.models.load_model(model_name)

    # Let's train the model using RMSprop
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6),
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics = ["accuracy",keras.metrics.sparse_top_k_categorical_accuracy])
    
    model.fit(x_train, y_train,
            epochs=epochs,
            batch_size=64,
            validation_split=0.1,
            shuffle=True,
            verbose=1)

    model.save(save_model)
    print('Saved trained model at %s ' % save_model)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])