import glob
from model import CNN
import numpy as np
import os
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import sys, os


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
model = CNN(patience=patience)
model.build_model(dataset_name, batch_size)
model.load_model("best_model.pkl")

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

# malignant recall
mal_dir = os.path.join(test_dir, 'malignant')
tp, total_mal = count_correct(mal_dir, 'malignant')
print(f"Correctly flagged {tp}/{total_mal} malignant images "
      f"({tp/total_mal*100:.1f}% recall)")

# benign recall
ben_dir = os.path.join(test_dir, 'benign')
tn, total_ben = count_correct(ben_dir, 'benign')
print(f"Correctly flagged {tn}/{total_ben} benign images "
      f"({tn/total_ben*100:.1f}% recall)")