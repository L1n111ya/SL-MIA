import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import ResNet50V2, ResNet101V2, DenseNet121, VGG16, VGG19


def get_Resnet50(input_shape, num_classes):
    """Define and return the ResNet50V2 model.
    Args:
        input_shape: a tensor defined the data shape.
        num_classes: the number of classes.
    Returns: 
        a keras model object named ResNet50V2.
    """
    model = tf.keras.Sequential([
        ResNet50V2(include_top=False,
                   weights='imagenet',
                   input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    
    return model


def get_ResNet101(input_shape, num_classes):
    """Define and return the ResNet101V2 model.
    Args:
        input_shape: a tensor defined the data shape.
        num_classes: the number of classes.
    Returns: 
        a keras model object named ResNet101V2.
    """
    model = tf.keras.Sequential([
        ResNet101V2(include_top=False,
                   weights='imagenet',
                   input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    
    return model


def get_DenceNet121(input_shape, num_classes):
    """Define and return the DenceNet121 model.
    Args:
        input_shape: a tensor defined the data shape.
        num_classes: the number of classes.
    Returns: 
        a keras model object named DenceNet121.
    """
    model = tf.keras.Sequential([
        DenseNet121(include_top=False,
                    weights='imagenet',
                    input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    
    return model


def get_VGG16(input_shape, num_classes):
    """Define and return the VGG16 model.
    Args:
        input_shape: a tensor defined the data shape.
        num_classes: the number of classes.
    Returns: 
        a keras model object named VGG16.
    """
    model = tf.keras.Sequential([
        VGG16(include_top=False,
                weights='imagenet',
                input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    
    return model


def get_VGG19(input_shape, num_classes):
    """Define and return the VGG19 model.
    Args:
        input_shape: a tensor defined the data shape.
        num_classes: the number of classes.
    Returns: 
        a keras model object named VGG19.
    """
    model = tf.keras.Sequential([
        VGG19(include_top=False,
                weights='imagenet',
                input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    
    return model


def get_CNN(input_shape, num_classes):
    """Define and return the CNN model.
    Args:
        input_shape: a tensor defined the data shape.
        num_classes: the number of classes.
    Returns: 
        a keras model object of CNN.
    """
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)), 
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    
    return model
