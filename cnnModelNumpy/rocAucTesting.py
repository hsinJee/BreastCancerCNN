import glob
from Main.model import CNN
import numpy as np
import os
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import sys, os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


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

print('\n--- Building the model ---')  
model = CNN(patience=patience)
model.build_model(dataset_name, batch_size)
model.load_model("best_model70b90m.pkl") 

class_names = ['benign','malignant']

def count_correct(dir_path, target_label):
    paths = glob.glob(os.path.join(dir_path, '*.png'))
    correct = 0
    for fp in paths:
        img = Image.open(fp).convert('RGB').resize((224,224))
        x   = np.expand_dims(np.array(img), axis=0)
        x   = preprocess_input(x)
        probs = model.predict(x)[0]
        pred  = class_names[np.argmax(probs)]
        if pred == target_label:
            correct += 1
    return correct, len(paths)


mal_dir = os.path.join(test_dir, 'malignant')
tp, total_mal = count_correct(mal_dir, 'malignant')
print(f"Correctly flagged {tp}/{total_mal} malignant images "
      f"({tp/total_mal*100:.1f}% recall)")

# benign recall
ben_dir = os.path.join(test_dir, 'benign')
tn, total_ben = count_correct(ben_dir, 'benign')
print(f"Correctly flagged {tn}/{total_ben} benign images "
      f"({tn/total_ben*100:.1f}% recall)")

y_true = []
y_scores = []

def collect_scores(dir_path, label_value):
    paths = glob.glob(os.path.join(dir_path, '*.png'))
    for fp in paths:
        img = Image.open(fp).convert('RGB').resize((224,224))
        x = np.expand_dims(np.array(img), axis=0)
        x = preprocess_input(x)
        probs = model.predict(x)[0]  
        y_scores.append(probs[1])  
        y_true.append(label_value) 


collect_scores(os.path.join(test_dir, 'benign'), label_value=0)
collect_scores(os.path.join(test_dir, 'malignant'), label_value=1)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Custom CNN')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()