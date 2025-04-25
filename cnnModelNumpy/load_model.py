from model_new import CNN
import numpy as np
import sys, os
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from loading import load_mnist_new, preprocess

dataset_name = 'mnist'
epochs = 1
learning_rate = 0.01
validate = 1
regularization = 0
verbose = 1
plot_weights = 1
batch_size = 32
patience = 5
regularization = 0.1

print('\n--- Loading ' + dataset_name + ' dataset ---')  
dataset = load_mnist_new()

print('\n--- Processing the dataset ---')  
dataset = preprocess(dataset)

print('\n--- Building the model ---')                                   # build model
model = CNN()
model.build_model(dataset_name, batch_size)
model.load_model("best_modelMNIST.pkl")

test_images = dataset['test_images']
test_labels = dataset['test_labels']

print("hello")
plt.imshow(test_images[0].reshape(28, 28), cmap='gray')
plt.title(f"Label: {test_labels[0]}")

# predict the number 9
target_digit = 2
indices = np.where(dataset['test_labels'] == target_digit)[0]
if len(indices) == 0:
    print(f"No images of digit {target_digit} found.")
else:
    first_index = indices[0]
    image = dataset['test_images'][first_index]

    # If your image is flattened (784), reshape it for viewing
    if image.ndim == 1:
        image = image.reshape(28, 28)

    # Display the image
    plt.imshow(image, cmap='gray')
    plt.title(f"Digit: {target_digit}")
    plt.axis('off')
    plt.show()

    probs = model.predict(image)

    print("\nClass probabilities:")
    for i, p in enumerate(probs[0]):
        print(f"Digit {i}: {p:.4f}")

