import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, log_loss
from glob import glob
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from loading import load_breakHis
from plotting import plot_accuracy_curve

IMAGE_SIZE=[224, 224]

train_dir = r"C:\Users\sumhs\Documents\Projects\BreastCancer\dataset_split2_200X\train"
val_dir = r"C:\Users\sumhs\Documents\Projects\BreastCancer\dataset_split2_200X\val"
test_dir = r"C:\Users\sumhs\Documents\Projects\BreastCancer\dataset_split2_200X\test"

base_model = ResNet50(input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False)
print(base_model.summary())

for layer in base_model.layers[:-20]:
    layer.trainable = False # Freeze all layers
for layer in base_model.layers[-20:]:
    layer.trainable = True # Unfreeze some layers

classes = glob(os.path.join(train_dir, "*"))

print(classes) # print number of classes
class_num = len(classes)

# build the model
global_avg_pooling_layer = GlobalAveragePooling2D()(base_model.output)

x = Dropout(0.5)(global_avg_pooling_layer)

# FlattenLayer = Flatten()(x)

# add the last layer
predictionLayer = Dense(class_num, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictionLayer)

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=['accuracy'])

# loading the files and data augmentation
training_set, val_set = load_breakHis(train_dir, val_dir)

EPOCHS = 200

best_model_file = r"C:\Users\sumhs\Documents\Projects\FYP\temp\BreakHis-ResNet50.keras"

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_accuracy', patience=10, verbose=1, factor=0.5, min_lr=0.00001),
    EarlyStopping(monitor='val_accuracy', patience=30, verbose=1, mode='max')
]

class_labels = training_set.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_labels), y=class_labels)

class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights: ", class_weights_dict)

r = model.fit(
    training_set,
    validation_data=val_set,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=len(training_set),
    validation_steps=len(val_set),
    class_weight=class_weights_dict  # added class weights to the model.fit call.
)

best_val_acc = max(r.history['val_accuracy'])

plot_accuracy_curve(r.history['accuracy'], r.history['val_accuracy'])
