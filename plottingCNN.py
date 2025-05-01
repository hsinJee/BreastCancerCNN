import pandas as pd
import matplotlib.pyplot as plt

def moving_average(data, window_size=20):
    return data.rolling(window=window_size, min_periods=1).mean()
    return data
# Path to the CSV history file
csv_path = r"C:\Users\sumhs\Documents\Projects\FYP\temp\history_data_vgg.csv"
csv_path =r"training_history_20250430_150000.csv"
# Load CSV
history_df = pd.read_csv(csv_path)

# Plot accuracy
plt.figure(figsize=(6, 4))
plt.plot(moving_average(history_df['accuracy']), label='Train Accuracy')
plt.plot(moving_average(history_df['val_accuracy']), label='Val Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Custom CNN Accuracy')
plt.legend()

# Plot loss
plt.figure(figsize=(6, 4))
plt.plot(moving_average(history_df['loss']), label='Train Loss')
plt.plot(moving_average(history_df['val_loss']), label='Val Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Custom CNN Loss')
plt.legend()

plt.tight_layout()
plt.show()


