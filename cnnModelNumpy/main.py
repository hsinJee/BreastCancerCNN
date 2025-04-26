from model import CNN
import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from loading import load_mnist_new, preprocess, load_breakHis_CNN

# exploding gradient for now
if __name__ == '__main__':
    # 'breakHis' 'mnist'
    dataset_name = 'breakHis'
    epochs = 5
    learning_rate = 0.01
    validate = 1
    regularization = 0
    verbose = 1
    plot_weights = 1
    batch_size = 32
    patience = 5
    regularization = 0.1
    train_dir = r"C:\Users\sumhs\Documents\Projects\BreastCancer\dataset_split2_200X\train"
    val_dir = r"C:\Users\sumhs\Documents\Projects\BreastCancer\dataset_split2_200X\val"
    test_dir = r"C:\Users\sumhs\Documents\Projects\BreastCancer\dataset_split2_200X\test"


    print('\n--- Loading ' + dataset_name + ' dataset ---')  
    dataset = load_breakHis_CNN(train_dir=train_dir, val_dir=val_dir, test_dir=test_dir)
    # dataset = load_mnist_new()
    print('\n--- Processing the dataset ---')  
    # dataset = preprocess(dataset)

    print(f'Length of train dataset: {len(dataset["train_images"])}')

    print('\n--- Building the model ---')                                   # build model
    model = CNN(patience=patience)
    model.build_model(dataset_name, batch_size)

    print('\n--- Training the model ---')
    model.train(
        dataset=dataset,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        regularization=regularization,
        validate=True
    )

    print('\n--- Evaluating the model ---')
    model.evaluate(
        X = dataset['validation_images'],
        y = dataset['validation_labels'],
        batch_size=batch_size,
        regularization=regularization,
        verbose=verbose
    )
