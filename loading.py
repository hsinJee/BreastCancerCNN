import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
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

def load_breakHis_vgg(train_dir, val_dir, image_size=(224, 224), batch_size=64):
    # augment the image for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_vgg,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # augment the image for validation
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_vgg  
    )

    training_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_set = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    return training_set, val_set

def load_breakHis_resNet(train_dir, val_dir, image_size=(224, 224), batch_size=64):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_resnet,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_resnet
    )

    training_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_set = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical'
    )
    return training_set, val_set

def load_breakHis_CNN(train_dir, val_dir, test_dir, image_size=(224, 224), batch_size=64):
    # Image data generator for training with augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_vgg,  # ResNet-specific preprocessing
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False
    )

    # Image data generator for validation (no augmentation)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_vgg 
    )
    
    # Image data generator for test set (no augmentation)
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_vgg  
    )

    # Load the training dataset
    train_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'  
    )

    # Load the validation dataset
    val_set = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False,  
        class_mode='categorical'
    )

    # Load the test dataset
    test_set = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False,  
        class_mode='categorical'
    )

    # Collect all images and labels
    def collect_data(data_generator):
        images = []
        labels = []
        for _ in range(len(data_generator)):
            batch_images, batch_labels = data_generator.next()
            images.append(batch_images)
            labels.append(batch_labels)
        return np.concatenate(images), np.concatenate(labels)
    
    # Get training, validation, and test data
    train_images, train_labels = collect_data(train_set)
    validation_images, validation_labels = collect_data(val_set)
    test_images, test_labels = collect_data(test_set)
    
    # Return the dictionary in the desired format
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