import seaborn as sns
import matplotlib.pyplot as plt

def plot_accuracy_curve(accuracy_history, val_accuracy_history):
    plt.plot(accuracy_history, 'b', linewidth=3.0, label='Training accuracy')
    plt.plot(val_accuracy_history, 'r', linewidth=3.0, label='Validation accuracy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy rate', fontsize=16)
    plt.legend()
    plt.title('Training Accuracy', fontsize=16)
    plt.savefig('training_accuracy.png')
    plt.show()


def plot_learning_curve(loss_history):
    plt.plot(loss_history, 'b', linewidth=3.0, label='Cross entropy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.title('Learning Curve', fontsize=16)
    plt.savefig('learning_curve.png')
    plt.show()


def plot_sample(image, true_label, predicted_label):
    plt.imshow(image)
    if true_label and predicted_label is not None:
        if type(true_label) == 'int':
            plt.title('True label: %d, Predicted Label: %d' % (true_label, predicted_label))
        else:
            plt.title('True label: %s, Predicted Label: %s' % (true_label, predicted_label))
    plt.show()


def plot_histogram(layer_name, layer_weights):
    plt.hist(layer_weights)
    plt.title('Histogram of ' + str(layer_name))
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.show()
