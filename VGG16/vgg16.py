import os
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import BatchNormalization 
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from glob import glob
import numpy as np
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from loading import load_breakHis_vgg


# using gpu for training
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    physical_devices[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # set memory limit to 4GB
)

IMAGE_SIZE = [224, 224]

train_dir = r"traindir" # get directories from breakhis dataset
val_dir = r"valdir"
test_dir = r"testdir"

# load base VGG16
base_model = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# freeze all VGG layers
for layer in base_model.layers[:-4]: # freezing the convolutional layers
    layer.trainable = False

print(base_model.summary())

classes = glob(os.path.join(train_dir, "*"))
class_num = len(classes)

print(f"Numer of classes: {class_num}")

# Build the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = BatchNormalization()(x)
predictions = Dense(class_num, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=['accuracy'])

# loading the files and data augmentation
training_set, val_set = load_breakHis_vgg(train_dir, val_dir)

# compute class weights
class_labels = training_set.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_labels), y=class_labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights: ", class_weights_dict)

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

best_model_file = r"C:\Users\sumhs\Documents\Projects\FYP\temp\BreakHis-VGG16.keras"
callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=1e-5),
    EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max')
]

EPOCHS = 200
history = model.fit(
    training_set,
    validation_data=val_set,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=len(training_set),
    validation_steps=len(val_set),
    class_weight=class_weights_dict
)

# save data into a csv file
history_df = pd.DataFrame(history.history)
csv_file = r"C:\Users\sumhs\Documents\Projects\FYP\temp\history_data_vgg.csv"
history_df.to_csv(csv_file, index=False)

print(f"History data saved to {csv_file}")

# Plot accuracy curve and loss curves
plot_accuracy_curve(history.history['accuracy'], history.history['val_accuracy'])
plot_learning_curve(history.history['loss'])

y_true = val_set.classes  # True class indices (e.g., 0 or 1)
y_pred_probs = model.predict(val_set)
y_pred = np.argmax(y_pred_probs, axis=1)  # Pick the class with highest probability
f1 = f1_score(y_true, y_pred, average='binary')

