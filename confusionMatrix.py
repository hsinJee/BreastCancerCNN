import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_res
import os

# Paths
IMAGE_SIZE = 224
BATCH_SIZE = 32
test_dir = r"test dir"


# Load model
best_model_file = r"C:\BreakHis-ResNet50.keras"
model = tf.keras.models.load_model(best_model_file)

# Prepare test data
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_res)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # IMPORTANT: ensures label order matches predictions
)

# Get true labels
y_true = test_generator.classes

# Predict
predictions = model.predict(test_generator, verbose=1)
y_pred_labels = np.argmax(predictions, axis=1)

# Get class labels
class_labels = list(test_generator.class_indices.keys())

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

# Plot
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix - ResNet50")
plt.tight_layout()
plt.show()
