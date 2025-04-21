from conv_new import Convolutional
from dense_new import Dense
from flatten import Flatten
import numpy as np
import time

class CNN:
    def __init__(self): 
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def build_model(self, dataset_name, batch_size):
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
            self.add_layer(Flatten()) # add flatten layer before dense
            self.add_layer(Dense(name='dense1', 
                        input_size=8*6*6,
                        output_size=10)) # 10 class for MNIST
            
    def forward(self, image):
        for layer in self.layers:
            image = layer.forward(image)
        return image
    
    def backward(self, dout, learning_rate):
        for layer in reversed(self.layers):
            dout = layer.backward(dout, learning_rate)     
        return dout
    
    def cross_entropy(self, predictions, targets, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        batch_size = predictions.shape[0]
        log_likelihood = -np.log(predictions[np.arange(batch_size), targets])
        return np.mean(log_likelihood)

    def regularized_cross_entropy(self, layers, predictions, targets, lam, batch_size):
        data_loss = self.cross_entropy(predictions, targets)
        
        reg_loss = 0
        for layer in layers:
            weights = layer.get_weights()
            if isinstance(weights, np.ndarray) and weights.size > 0:  # Skip layers without weights
                reg_loss += np.sum(weights**2)
        
        total_loss = data_loss + (lam / (2 * batch_size)) * reg_loss
        return total_loss
    
    def train(self, dataset, num_epochs, batch_size, learning_rate, regularization):
        print(f"Starting training for {num_epochs} epochs with batch size {batch_size}...")    

        # do not care about val_loss val_accuracy and lr for now ill implement later
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}
        X_train = dataset['train_images']
        y_train = dataset['train_labels']

        n_train = len(X_train)

        for epoch in range(num_epochs):
            epoch_loss, epoch_correct = 0, 0
            num_batches = X_train.shape[0] // batch_size

            indices = np.arange(n_train)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            initial_time = time.time()

            for i in range(num_batches):
                start = i * batch_size # take the ith batch
                end = start + batch_size 
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]
                
                # pass batch through forward pass
                y_pred = self.forward(X_batch)

                # number of correct preds
                batch_preds = np.argmax(y_pred, axis=1)
                batch_correct = np.sum(batch_preds == y_batch)

                epoch_correct += batch_correct
                batch_accuracy = batch_correct / batch_size
                num_classes = y_pred.shape[1]

                # backward pass
                gradient = y_pred.copy()
                gradient[np.arange(len(X_batch)), y_batch] -= 1
                gradient /= len(X_batch)

                
                # loss function
                loss = self.regularized_cross_entropy(self.layers, y_pred, y_batch, regularization, len(X_batch))
                epoch_loss += loss
                
                self.backward(gradient, learning_rate)

                history['loss'].append(loss)
                history['accuracy'].append(batch_accuracy)

                # print each first and last batch and every 50 batches
                if (i + 1) % 50 == 0 or i == 1 or i == num_batches:
                    print(f"Batch {i+1}/{num_batches}, Loss: {loss:.4f}, Accuracy: {batch_accuracy:.4f}, Time: {(time.time() - initial_time):.2f}s")

                    # restart time
                    initial_time = time.time()

            # average loss for this batch
            epoch_loss /= num_batches
            accuracy = epoch_correct / (num_batches * batch_size)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {accuracy}, Time: {time.time() - initial_time}")
            

            