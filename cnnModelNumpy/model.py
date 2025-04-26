from conv import Convolutional
from dense import Dense
from flatten import Flatten
from batch_norm import BatchNormalization
from pooling import Pooling
import numpy as np
import time
import pickle
import pandas as pd
import datetime

class CNN:
    def __init__(self, patience): 
        self.layers = []
        self.training = True
        self.patience = patience
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def build_model(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        if dataset_name == 'mnist':
            self.add_layer(Convolutional(name='conv1', 
                                    image_shape=(28, 28, 1),
                                    num_filters=8,
                                    stride=2,
                                    size=3,  
                                    activation='relu'))
            self.add_layer(Convolutional(name='conv2',
                                        image_shape=(13, 13, 8),  # Output shape from conv1
                                        num_filters=8,
                                        stride=2,
                                        size=3,
                                        activation='relu'))
            self.add_layer(Flatten(name='flatten1')) # add flatten layer before dense
            self.add_layer(Dense(name='dense1', 
                        input_size=8*6*6,
                        output_size=10, 
                        useSoftmax=True)) # use softmax as it is the final layer
            
        elif dataset_name == 'other':
            self.add_layer(Convolutional(name='conv1', 
                                    image_shape=(28, 28, 1),
                                    num_filters=8,
                                    stride=2,
                                    size=3,  
                                    activation='relu'))
            self.add_layer(BatchNormalization(name='batch_norm1'))
            self.add_layer(Convolutional(name='conv2',
                                        image_shape=(13, 13, 8),  # Output shape from conv1
                                        num_filters=8,
                                        stride=2,
                                        size=3,
                                        activation='relu'))
            self.add_layer(BatchNormalization(name='batch_norm2'))
            self.add_layer(Flatten(name='flatten1')) # add flatten layer before dense
            self.add_layer(Dense(name='dense1', 
                        input_size=8*6*6,
                        output_size=10, 
                        useSoftmax=True)) # use softmax as it is the final layer    
                
        elif dataset_name == 'breakHis':
            self.add_layer(Convolutional(name='conv1', 
                              image_shape=(225, 225, 3),  # Input shape (225, 225, 3) for RGB
                              num_filters=16,
                              stride=2,
                              size=3,  # Kernel size
                              activation='relu'))
            # Batch Normalization 1
            self.add_layer(BatchNormalization(name='batch_norm1'))

            # Pooling Layer 1
            self.add_layer(Pooling(name='pool1', stride=2, size=2))  # Pooling with stride=2, size=2

            # Convolutional Layer 2
            self.add_layer(Convolutional(name='conv2',
                                        image_shape=(56, 56, 16),  # Output shape from conv1
                                        num_filters=32,
                                        stride=2,
                                        size=3,
                                        activation='relu'))

            # Batch Normalization 2
            self.add_layer(BatchNormalization(name='batch_norm2'))

            # Pooling Layer 2
            self.add_layer(Pooling(name='pool2', stride=2, size=2))

            # Convolutional Layer 3
            self.add_layer(Convolutional(name='conv3',
                                        image_shape=(13, 13, 32),  # Output shape from conv2
                                        num_filters=64,
                                        stride=2,
                                        size=3,
                                        activation='relu'))

            # Batch Normalization 3
            self.add_layer(BatchNormalization(name='batch_norm3'))

            # Pooling Layer 3
            self.add_layer(Pooling(name='pool3', stride=2, size=2))

            # Flatten Layer
            self.add_layer(Flatten(name='flatten1'))  # Flatten the output to pass into dense layers

            # Fully Connected Layer 1 (Dense Layer)
            self.add_layer(Dense(name='dense1', 
                                input_size= 3*3*64,  # Adjusted for the output size after pooling
                                output_size=128,  
                                useSoftmax=False))

            # Output Layer (Softmax for classification)
            self.add_layer(Dense(name='output', 
                                input_size=128, 
                                output_size=2,  # For binary classification (if there are two classes)
                                useSoftmax=True))
            
    def lr_scheduler(self, initial_lr, step, drop=0.5, epochs_drop=10):
        new_lr = initial_lr * (drop ** (step // epochs_drop))
        print(f"Adjusting learning rate: {initial_lr} > {new_lr}")
        return new_lr
            
    def forward(self, image, training):
        # training bool false or true
        for layer in self.layers:
            image = layer.forward(image, training)
            print(f"{layer.name}")
        return image
    
    def backward(self, dout, learning_rate):
        print("--backwards pass--")
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                dout = layer.backward(dout, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        return dout
    
    def cross_entropy(self, predictions, targets, class_weights=None, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        batch_size = predictions.shape[0]
        
        log_likelihood = -np.log(predictions[np.arange(batch_size), targets])

        if class_weights is not None:
            weights = class_weights[targets]
            log_likelihood *= weights

        return np.mean(log_likelihood)

    def regularized_cross_entropy(self, layers, predictions, targets, lam, batch_size, class_weights=None):
        data_loss = self.cross_entropy(predictions, targets, class_weights=class_weights)

        reg_loss = 0
        for layer in layers:
            weights = layer.get_weights()
            if isinstance(weights, np.ndarray) and weights.size > 0:
                reg_loss += np.sum(weights**2)

        total_loss = data_loss + (lam / (2 * batch_size)) * reg_loss
        return total_loss
    
    def train(self, dataset, num_epochs, batch_size, learning_rate, regularization, verbose=True, validate=True):
        print(f"Starting training for {num_epochs} epochs with batch size {batch_size}...")    

        history = {'loss': [], 'accuracy': [], 'lr': [], 'val_loss': [], 'val_accuracy': []}

        initial_learning_rate = learning_rate

        X_train = dataset['train_images']
        y_train = dataset['train_labels']

        n_train = len(X_train)

        n_batches = (n_train + batch_size - 1) // batch_size
        eval_interval = max(1, n_batches // 20) # evaluate 20 times per epoch

        best_val_accuracy = -np.inf
        interval_loss = 0
        interval_correct = 0
        interval_count = 0
        early_stopping = 0

        for epoch in range(num_epochs):
            self.training = True
            epoch_loss, epoch_correct = 0, 0

            indices = np.arange(n_train)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            initial_time = time.time()

            for i in range(0, n_train, batch_size):
                batch_num = i // batch_size + 1

                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                
                y_pred = self.forward(X_batch, training=True)

                batch_preds = np.argmax(y_pred, axis=1)

                if self.dataset_name == 'breakHis':
                    y_true = np.argmax(y_batch, axis=1)
                    batch_correct = np.sum(batch_preds == y_true)
                elif self.dataset_name == 'mnist': 
                    batch_correct = np.sum(batch_preds == y_batch)

                epoch_correct += batch_correct
                batch_accuracy = batch_correct / batch_size

                gradient = y_pred.copy()
                gradient[np.arange(len(X_batch)), y_batch] -= 1
                gradient /= len(X_batch)

                loss = self.regularized_cross_entropy(self.layers, y_pred, y_batch, regularization, len(X_batch))

                epoch_loss += loss

                interval_loss += loss
                interval_correct += batch_correct
                interval_count += 1

                self.backward(gradient, learning_rate)

                if batch_num % 50 == 0 or batch_num == 1 or batch_num == n_batches:
                    print(f"Batch {batch_num}/{n_batches}, Loss: {loss:.4f}, Accuracy: {batch_accuracy:.4f}, Time: {(time.time() - initial_time):.2f}s")
                    initial_time = time.time()
                

                if batch_num % eval_interval == 0 or batch_num == n_batches:
                    print("evaluating now")
                    avg_loss = interval_loss / interval_count
                    avg_accuracy = interval_correct / (interval_count * batch_size)

                    interval_step = (batch_num // eval_interval)
                    learning_rate = self.lr_scheduler(initial_learning_rate, interval_step, drop=0.5, epochs_drop=5)

                    history['lr'].append(learning_rate)
                    history['loss'].append(avg_loss)
                    history['accuracy'].append(avg_accuracy)

                    interval_loss = 0
                    interval_correct = 0
                    interval_count = 0

                    self.training = False
                    val_loss, val_accuracy = self.evaluate(
                        dataset['validation_images'],
                        dataset['validation_labels'],
                        batch_size,
                        regularization,
                        verbose,
                        fraction=0.3
                    )
                    self.training = True

                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_accuracy)

                    if val_accuracy > best_val_accuracy:
                        print(f'--New best validation accuracy: {best_val_accuracy:.4f} > {val_accuracy:.4f}. Saving model...--')
                        best_val_accuracy = val_accuracy
                        self.save_model()
                        early_stopping = 0
                    else:
                        early_stopping += 1
                        print(f'No improvemenst for {early_stopping} intervals.')
                        if early_stopping >= self.patience:
                            print(f'\nEarly stopping! No improvement for {self.patience} intervals.')
                            break

            # after full epoch - no extra evaluate here
            epoch_loss /= n_batches
            accuracy = epoch_correct / n_train
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {epoch_loss:.4f}, Avg Accuracy: {accuracy:.4f}")

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        history_df = pd.DataFrame(history)
        history_df.to_csv(f'training_history_{timestamp}.csv', index=False)

        print("Training and history saved.")

    def evaluate(self, X, y, batch_size, regularization, verbose=True, fraction=0.3):
        n_data = len(X)
        subset_size = int(fraction * n_data)

        if subset_size < batch_size:
            subset_size = batch_size

        indices = np.random.choice(n_data, subset_size, replace=False)
        X = X[indices]
        y = y[indices]

        n_data = len(X)
        num_batches = (n_data + batch_size - 1) // batch_size 
        total_loss = 0 
        total_correct = 0

        for i in range(0, n_data, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            y_pred = self.forward(X_batch, training=False)

            y_pred  = self.forward(X_batch, training=False)
            loss = self.regularized_cross_entropy(self.layers, y_pred, y_batch, regularization, len(X_batch))
            total_loss += loss

            preds = np.argmax(y_pred, axis=1)
            batch_correct = np.sum(preds == y_batch)
            total_correct += batch_correct
            batch_accuracy = batch_correct / len(X_batch)
            print(f"Evaluation Batch {i // batch_size + 1}/{num_batches}, Loss: {loss:.4f}, Accuracy: {batch_accuracy:.4f}")

        val_loss = total_loss / num_batches
        val_acc = total_correct / n_data

        return val_loss, val_acc
    
    def get_parameters(self):
        return  [layer.params for layer in self.layers if hasattr(layer, 'params')]
    
    def set_parameters(self, parameters):
        idx = 0
        for layer in self.layers:
            if hasattr(layer, 'params'):
                layer.params = parameters[idx]
                idx += 1

    def save_model(self, filename='best_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_model(self, filename='best_model.pkl'):
        with open(filename, 'rb') as f:
            params = pickle.load(f)
            self.set_parameters(params)

    def predict(self, image):
        # Ensure image is 4D: (1, 28, 28, 1)
        if image.ndim == 2:
            image = image.reshape(1, 28, 28, 1)
        elif image.ndim == 3:
            image = image.reshape(1, *image.shape)
        elif image.ndim != 4:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        print("Image shape:", image.shape)
        probs = self.forward(image, training=False)
        prediction = np.argmax(probs, axis=1)[0]
        print(f"Predicted digit: {prediction}")
        return probs
