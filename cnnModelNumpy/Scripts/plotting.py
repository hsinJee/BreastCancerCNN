import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your history
history = pd.read_csv('training_history_20250426_004337.csv')

# Choose columns
train_acc = history['accuracy']
val_acc = history['val_accuracy']

# Insert (0, 0) at the beginning
train_acc = np.insert(train_acc.values, 0, 0.0)
val_acc = np.insert(val_acc.values, 0, 0.0)

# Create x axis
x_real = np.arange(len(train_acc))  # Now one longer
x_dense = np.linspace(0, len(train_acc) - 1, len(train_acc) * 5)  # 5x more points

# Interpolate
train_acc_interp = np.interp(x_dense, x_real, train_acc)
val_acc_interp = np.interp(x_dense, x_real, val_acc)

# Plot
plt.figure(figsize=(10,5))
plt.plot(x_dense, train_acc_interp, label='Training Accuracy (smooth)', linestyle='-')
plt.plot(x_dense, val_acc_interp, label='Validation Accuracy (smooth)', linestyle='-')
plt.xlabel('Interval')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy (Smoothed, Starting from 0)')
plt.legend()
plt.grid(True)
plt.show()


history = pd.read_csv('training_history_20250425_172917.csv')

group_size = 100  # Average every 100 batches
grouped = history.groupby(history.index // group_size).mean()

# Plot smoothed Training Loss
plt.figure(figsize=(10,5))
plt.plot(grouped.index * group_size, grouped['loss'], label='Training Loss (avg every 100 batches)', linestyle='-')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss Averaged Every 100 Batches (SGD)')
plt.legend()
plt.grid(True)
plt.show()

# Plot smoothed Training Accuracy
plt.figure(figsize=(10,5))
plt.plot(grouped.index * group_size, grouped['accuracy'], label='Training Accuracy (avg every 100 batches)', linestyle='-')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Averaged Every 100 Batches (SGD)')
plt.legend()
plt.grid(True)
plt.show()

# Plot Learning Rate
plt.figure(figsize=(10,5))
plt.plot(grouped.index * group_size, grouped['lr'], label='Learning Rate (avg every 100 batches)', linestyle='-')
plt.xlabel('Batch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Averaged Every 100 Batches')
plt.legend()
plt.grid(True)
plt.show()