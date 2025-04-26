import tensorflow as tf
import numpy as np
def load_mnist():
    (X_train, train_labels), (X_test, test_labels) = tf.keras.datasets.mnist.load_data()

    # Reshape to (N, 1, 28, 28) to match original format
    train_images = np.expand_dims(X_train, axis=1)
    test_images = np.expand_dims(X_test, axis=1)

    # Permute and split into training and validation sets
    indices = np.random.permutation(train_images.shape[0])
    training_idx, validation_idx = indices[:55000], indices[55000:]
    train_images, validation_images = train_images[training_idx], train_images[validation_idx]
    train_labels, validation_labels = train_labels[training_idx], train_labels[validation_idx]

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

def load_mnist_new():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape to (N, 28, 28, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Permute and split
    indices = np.random.permutation(X_train.shape[0])
    training_idx, validation_idx = indices[:55000], indices[55000:]

    train_images = X_train[training_idx]
    train_labels = y_train[training_idx]
    validation_images = X_train[validation_idx]
    validation_labels = y_train[validation_idx]  

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'test_images': X_test,
        'test_labels': y_test
    }

def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x

def preprocess(dataset):
    dataset['train_images'] = np.array([minmax_normalize(x) for x in dataset['train_images']])
    dataset['validation_images'] = np.array([minmax_normalize(x) for x in dataset['validation_images']])
    dataset['test_images'] = np.array([minmax_normalize(x) for x in dataset['test_images']])
    return dataset