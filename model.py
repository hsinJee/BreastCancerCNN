from conv import Convolutional
from dense import Dense
from plotting import plot_accuracy_curve, plot_histogram, plot_learning_curve, plot_sample
import numpy as np
import time

def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    batch_size = predictions.shape[0]
    log_likelihood = -np.log(predictions[np.arange(batch_size), targets])
    return np.mean(log_likelihood)

def regularized_cross_entropy(layers, predictions, targets, lam, batch_size):
    data_loss = cross_entropy(predictions, targets)
    
    reg_loss = 0
    for layer in layers:
        weights = layer.get_weights()
        if isinstance(weights, np.ndarray) and weights.size > 0:  # Skip layers without weights
            reg_loss += np.sum(weights**2)
    
    total_loss = data_loss + (lam / (2 * batch_size)) * reg_loss
    return total_loss

def lr_schedule(learning_rate, iteration): 
    if iteration < 10000:
        return learning_rate
    elif iteration <= 30000:
        return learning_rate * 0.1
    else:
        return learning_rate * 0.01  # Reduce further if necessary

class CNN:
    def __init__(self): 
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def build_model(self, dataset_name, batch_size):
        if dataset_name == 'mnist':
            self.add_layer(Convolutional(name='conv1', 
                                   image_shape=(1, 28, 28),
                                   num_filters=8,
                                   stride=2,
                                   size=3,
                                   padding=0,  
                                   activation='relu'))
            self.add_layer(Convolutional(name='conv2',
                                    image_shape=(8, 13, 13),  # Output shape from conv1
                                    num_filters=8,
                                    stride=2,
                                    size=3,
                                    padding=0,
                                    activation='relu'))
            self.add_layer(Dense(name='dense',
                            nodes=8*6*6,  
                            num_classes=10))
    
    def forward(self, image):
        for layer in self.layers:
            image = layer.forward(image)
        return image
    
    def backward(self, gradient, learning_rate):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

    def train(self, dataset, epochs, learning_rate, validate, regularization, batch_size=32, plot_weights=False, verbose=True, patience=5):
        print(f"Starting training for {epochs} epochs with batch size {batch_size}...")
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        X_train = dataset['train_images']
        y_train = dataset['train_labels']
        n_train = len(X_train)
        n_batches = n_train // batch_size + (1 if n_train % batch_size != 0 else 0)

        # Early stopping params
        best_val_loss = float('inf')
        epochs_since_improvement = 0

        for epoch in range(1, epochs + 1):
            if verbose: 
                print('\n--- Epoch {0} ---'.format(epoch))
            
            # Shuffle data
            indices = np.arange(n_train)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss, epoch_correct = 0, 0
            initial_time = time.time()

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_train)
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]
                actual_batch_size = len(X_batch)

                # check x batch before passing it through
                if len(X_batch.shape) == 3:
                    X_batch = X_batch[:, np.newaxis, :, :]

                # Forward pass (whole batch)
                batch_outputs = self.forward(X_batch)
                
                # Calculate loss and accuracy
                batch_loss = regularized_cross_entropy(
                    self.layers, 
                    batch_outputs, 
                    y_batch, 
                    regularization, 
                    actual_batch_size
                )
                batch_preds = np.argmax(batch_outputs, axis=1)
                batch_correct = np.sum(batch_preds == y_batch)

                # Backward pass
                gradient = batch_outputs.copy()
                gradient[np.arange(actual_batch_size), y_batch] -= 1
                gradient /= actual_batch_size  # Normalize by batch size
                
                # Update learning rate (per batch)
                global_iter = epoch * n_batches + batch_idx
                current_lr = lr_schedule(learning_rate, iteration=global_iter)
                
                self.backward(gradient, current_lr)

                # Update metrics
                epoch_loss += batch_loss * actual_batch_size
                epoch_correct += batch_correct

                # Logging and validation 
                if batch_idx % 50 == 0 or batch_idx == n_batches - 1:
                    avg_loss = epoch_loss / (batch_idx * batch_size + actual_batch_size)
                    accuracy = epoch_correct / (batch_idx * batch_size + actual_batch_size)
                    
                    history['loss'].append(avg_loss)
                    history['accuracy'].append(accuracy)
                    
                    if validate:
                        val_loss, val_accuracy = self.evaluate(
                            dataset['validation_images'],
                            dataset['validation_labels'],
                            regularization,
                            batch_size,
                            verbose=False
                        )
                        history['val_loss'].append(val_loss)
                        history['val_accuracy'].append(val_accuracy)
                        
                        if verbose:
                            print(f'[Batch {batch_idx+1}/{n_batches}]: '
                                f'Loss {avg_loss:.3f} | Acc: {accuracy:.3f} | Time: {(time.time() - initial_time):.3f} | '
                                f'Val Loss {val_loss:.3f} | Val Acc: {val_accuracy:.3f}')
                        
                        # Early stopping check
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            epochs_since_improvement = 0
                        else:
                            epochs_since_improvement += 1

                        if epochs_since_improvement >= patience:
                            print(f"Early stopping triggered after {epoch} epochs")
                            break
                    # restart time 
                    initial_time = time.time()
            # Early stopping check at epoch level
            if epochs_since_improvement >= patience:
                break

        # Final outputs
        if verbose:
            print('Final Training Loss: %.3f' % history['loss'][-1])
            print('Final Training Accuracy: %.3f' % history['accuracy'][-1])
            plot_learning_curve(history['loss'])
            plot_accuracy_curve(history['accuracy'], history['val_accuracy'])

        if plot_weights:
            for layer in self.layers:
                if 'pool' not in layer.name:
                    plot_histogram(layer.name, layer.get_weights())


    def evaluate(self, X, y, regularization, batch_size, verbose=True):
        n_data = len(X)
        total_loss = 0
        total_correct = 0
        
        for i in range(0, n_data, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            batch_size_actual = len(X_batch)

            # Forward pass (whole batch)
            batch_outputs = self.forward(X_batch)
            
            # Calculate loss and accuracy
            batch_loss = regularized_cross_entropy(
                self.layers, 
                batch_outputs, 
                y_batch, 
                regularization, 
                batch_size_actual
            )
            batch_preds = np.argmax(batch_outputs, axis=1)
            batch_correct = np.sum(batch_preds == y_batch)

            # Update metrics
            total_loss += batch_loss * batch_size_actual
            total_correct += batch_correct

        avg_loss = total_loss / n_data
        accuracy = total_correct / n_data
        
        if verbose:
            print('Evaluation Loss: %.3f' % avg_loss)
            print('Evaluation Accuracy: %.3f' % accuracy)
            
        return avg_loss, accuracy
