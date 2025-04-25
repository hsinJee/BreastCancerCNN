import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_history_20250425_172917.csv')

# Assign group numbers: 0 for first 100 rows, 1 for next 100, etc.
grouped = df.groupby(df.index // 100)

# Take the mean for each group
df_avg = grouped.mean().reset_index()

plt.figure(figsize=(10, 5))
plt.plot(df_avg.index * 100, df_avg['loss'], label='Avg Loss (per 100 batches)', color='red')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Average Loss Every 100 Batches')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df_avg.index * 100, df_avg['accuracy'], label='Avg Accuracy (per 100 batches)', color='blue')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.title('Average Accuracy Every 100 Batches')
plt.grid(True)
plt.legend()
plt.show()