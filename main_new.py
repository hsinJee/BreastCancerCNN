from model_new import CNN
from loading import load_mnist_new, preprocess

# exploding gradient for now
if __name__ == '__main__':

    dataset_name = 'mnist'
    epochs = 1
    learning_rate = 0.0001
    validate = 1
    regularization = 0
    verbose = 1
    plot_weights = 1
    batch_size = 32
    patience = 5


    print('\n--- Loading ' + dataset_name + ' dataset ---')  
    dataset = load_mnist_new()

    print('\n--- Processing the dataset ---')  
    dataset = preprocess(dataset)

    print(f'Length of train dataset: {len(dataset["train_images"])}')

    print('\n--- Building the model ---')                                   # build model
    model = CNN()
    model.build_model(dataset_name, batch_size)

    print('\n--- Training the model ---')
    model.train(
        dataset=dataset,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
