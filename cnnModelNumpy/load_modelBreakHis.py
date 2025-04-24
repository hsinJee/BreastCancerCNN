from model import CNN
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from loading import load_mnist_new, preprocess, load_breakHis_CNN

dataset_name = 'breakHis'
epochs = 1
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

print('\n--- Building the model ---')  # Build the model
model = CNN()
model.build_model(dataset_name, batch_size)
model.load_model("best_model.pkl")

# Load the image from the path
img_path = r"C:\Users\sumhs\Documents\Projects\BreastCancer\dataset_split2_200X\test\benign\SOB_B_F-14-25197-200-040.png"
image = Image.open(img_path).convert("RGB")
image = image.resize((224, 224))


# Convert to NumPy array
image = np.array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)


# Predict using the model
probs = model.predict(image)

# Print predicted class probabilities
print("\nPredicted class probabilities:")
for i, p in enumerate(probs[0]):
    print(f"Class {i}: {p:.4f}")

# Get the predicted class (assuming binary classification with 'benign' and 'malignant')
class_names = ['benign', 'malignant']
predicted_class = class_names[np.argmax(probs)]
print(f"\nPredicted class: {predicted_class}")

image = Image.open(img_path).convert("RGB")
image = image.resize((224, 224))

# Plot the image for verification
plt.imshow(image)
plt.title(f"Predicted class: {predicted_class}")
plt.axis('off')  # Hide axes
plt.show()