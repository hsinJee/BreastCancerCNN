from model import CNN
from loading import load_mnist, preprocess
import numpy as np

if __name__ == '__main__':

    dataset_name = 'mnist'
    epochs = 1
    learning_rate = 0.01
    validate = 1
    regularization = 0
    verbose = 1
    plot_weights = 1
    batch_size = 32
    patience = 5


    print('\n--- Loading ' + dataset_name + ' dataset ---')  
    dataset = load_mnist()

    print('\n--- Processing the dataset ---')  
    dataset = preprocess(dataset)

    print(f'Length of train dataset: {len(dataset["train_images"])}')

    print('\n--- Building the model ---')                                   # build model
    model = CNN()
    model.build_model(dataset_name, batch_size)

    print('\n--- Training the model ---')
    model.train(
        dataset,
        epochs,
        learning_rate,
        validate,
        regularization,
        batch_size,
        plot_weights,
        verbose,
        patience
    )
