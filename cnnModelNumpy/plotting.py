import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


history = pd.read_csv('training_history_20250426_004337.csv')


train_acc = history['accuracy']
val_acc = history['val_accuracy']


train_acc = np.insert(train_acc.values, 0, 0.0)
val_acc = np.insert(val_acc.values, 0, 0.0)

total_batches = 1719

intervals = len(train_acc) - 1
batches_per_interval = total_batches / intervals

x_real_batches = np.arange(len(train_acc)) * batches_per_interval


x_dense_batches = np.linspace(0, total_batches, len(train_acc) * 5)


train_acc_interp = np.interp(x_dense_batches, x_real_batches, train_acc)
val_acc_interp   = np.interp(x_dense_batches, x_real_batches, val_acc)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(x_dense_batches, train_acc_interp, label='Training Accuracy')
ax.plot(x_dense_batches, val_acc_interp,   label='Validation Accuracy')


ax.set_xlim(0, total_batches)
ax.set_ylim(0, 1.0)


ax.margins(x=0, y=0)


ax.set_xticks(np.arange(0, total_batches+1, 200))
ax.set_xlabel('Batch Number')
ax.set_ylabel('Accuracy')
ax.set_title('Training and Validation Accuracy (ADAM)')
ax.legend()
ax.grid(True)

plt.show()


history = pd.read_csv('training_history_20250425_172917.csv')

group_size = 100 
grouped = history.groupby(history.index // group_size).mean()


plt.figure(figsize=(10,5))
plt.plot(grouped.index * group_size, grouped['loss'], label='Training Loss (avg every 100 batches)', linestyle='-')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss Averaged Every 100 Batches (SGD)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10,5))
plt.plot(grouped.index * group_size, grouped['accuracy'], label='Training Accuracy (avg every 100 batches)', linestyle='-')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Averaged Every 100 Batches (SGD)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10,5))
plt.plot(grouped.index * group_size, grouped['lr'], label='Learning Rate (avg every 100 batches)', linestyle='-')
plt.xlabel('Batch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Averaged Every 100 Batches')
plt.legend()
plt.grid(True)
plt.show()