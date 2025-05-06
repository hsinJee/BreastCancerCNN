import glob
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications.vgg16 import preprocess_input

from Main.model import CNN

# Configuration
test_dir = r"C:\Users\sumhs\Documents\Projects\BreastCancer\dataset_split2_200X\test"
class_names = ['benign', 'malignant']
model_path = "best_model70b90m.pkl"

# Load model
model = CNN(patience=5)
model.build_model('breakHis2', batch_size=32)
model.load_model(model_path)

# Gather predictions and ground truth
y_true = []
y_pred = []

for label in class_names:
    dir_path = os.path.join(test_dir, label)
    image_paths = glob.glob(os.path.join(dir_path, '*.png'))

    for fp in image_paths:
        img = Image.open(fp).convert('RGB').resize((224, 224))
        x = np.expand_dims(np.array(img), axis=0)
        x = preprocess_input(x)
        probs = model.predict(x)[0]
        pred_class = class_names[np.argmax(probs)]

        y_true.append(label)
        y_pred.append(pred_class)

# Compute and plot confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=class_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Custom CNN")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
