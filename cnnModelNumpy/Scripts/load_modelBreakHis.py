from Main.model import CNN
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
import time

start = time.time()

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from loading import load_mnist_new, preprocess, load_breakHis_CNN

dataset_name = 'breakHis2'
epochs = 1
learning_rate = 0.01
validate = 1
regularization = 0
verbose = 1
plot_weights = 1
batch_size = 32
patience = 5
regularization = 0.1
train_dir = r"traindir" # get directories from breakhis dataset
val_dir = r"valdir"
test_dir = r"testdir"

print('\n--- Building the model ---')  # Build the model
model = CNN(patience=patience)
model.build_model(dataset_name, batch_size)
model.load_model("best_model70b90m.pkl")

# Load the image from the path
img_path = r"sample image path"
image = Image.open(img_path).convert("RGB")
image = image.resize((224, 224))


# Convert to NumPy array
image = np.array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)


# Predict using the model
probs = model.predict(image)
feature_maps = model.get_feature_maps(image)

# Print predicted class probabilities
print("\nPredicted class probabilities:")
for i, p in enumerate(probs[0]):
    print(f"Class {i}: {p:.4f}")

# Get the predicted class 
class_names = ['benign', 'malignant']
predicted_class = class_names[np.argmax(probs)]
print(f"\n Actual class: Malignant, Predicted class: {predicted_class}")

image = Image.open(img_path).convert("RGB")
image = image.resize((224, 224))
end = time.time()

plt.imshow(image)
plt.title(f"Actual class: malignant, Predicted class: {predicted_class}")
plt.axis('off')  
plt.show()

output_dir = r"C:\Users\sumhs\Documents\Projects\FYP\featuremaps"
os.makedirs(output_dir, exist_ok=True)

for name, fmap in feature_maps:
    fmap = np.squeeze(fmap)
    num_filters = fmap.shape[-1]

    n_display = min(num_filters, 32)
    fig , axes = plt.subplots(1, n_display, figsize=(20, 5))
    for i in range(n_display):
        fig, ax = plt.subplots(figsize=(2, 2))  # small size per map
        ax.imshow(fmap[:, :, i], cmap='viridis')
        ax.axis('off')

        filename = f"{name}_map{i+1}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # 


print("Inference time:", (end - start), "seconds")