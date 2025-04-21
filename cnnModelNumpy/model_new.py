from conv_new import Convolutional
from dense_new import Dense
from flatten import Flatten
import numpy as np
import time
import pickle

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
    
    def train(self, dataset, num_epochs, batch_size, learning_rate, regularization, verbose=True, validate=True):
        print(f"Starting training for {num_epochs} epochs with batch size {batch_size}...")    

        # do not care about val_loss val_accuracy and lr for now ill implement later
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}
        X_train = dataset['train_images']
        y_train = dataset['train_labels']

        n_train = len(X_train)
        
        best_accuracy = 0
        best_val_accuracy = -np.inf # start with very low value

        for epoch in range(num_epochs):
            epoch_loss, epoch_correct = 0, 0
            num_batches = (n_train + batch_size - 1) // batch_size

            indices = np.arange(n_train)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            initial_time = time.time()

            # for evaluation, only evaluate the last 70% of batches
            threshold_batch = int(num_batches * 0.7)

            for i in range(0, n_train, batch_size): 
                batch_num = i // batch_size + 1

                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                
                # pass batch through forward pass
                y_pred = self.forward(X_batch)

                # number of correct preds
                batch_preds = np.argmax(y_pred, axis=1)
                batch_correct = np.sum(batch_preds == y_batch)

                epoch_correct += batch_correct
                batch_accuracy = batch_correct / batch_size

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
                if batch_num % 50 == 0 or batch_num == 1 or batch_num == num_batches:
                    print(f"Batch {batch_num}/{num_batches}, Loss: {loss:.4f}, Accuracy: {batch_accuracy:.4f}, Time: {(time.time() - initial_time):.2f}s")
                    # restart time
                    initial_time = time.time()

                if batch_accuracy > best_accuracy:
                    print(f"Batch {batch_num}/{num_batches}, Loss: {loss:.4f}, Accuracy increase {best_accuracy} > {batch_accuracy}")
                    best_accuracy = batch_accuracy

                    if validate and batch_num >= threshold_batch: # only start evaluating after threshold
                        best_val_accuracy = self.call_evaluate(self, dataset, history, batch_size, best_val_accuracy, regularization, verbose)
            

            # evaluate after epoch 
            best_val_accuracy = self.call_evaluate(self, dataset, history, batch_size, best_val_accuracy, regularization, verbose)
            # average loss for this batch
            epoch_loss /= num_batches
            accuracy = epoch_correct / n_train
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {epoch_loss}, Avg Accuracy: {accuracy}, Best Accuracy: {best_accuracy}")

    def call_evaluate(self, dataset, history, batch_size, best_val_accuracy, regularization, verbose):
        # evaluate after epoch 
        indices = np.random.permutation(dataset['validation_images'].shape[0])
        val_loss, val_accuracy = self.evaluate(
            dataset['validation_images'][indices, :],
            dataset['validation_labels'][indices],
            batch_size,
            regularization,
            verbose
        )
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            print(f'--New best validation accuracy: {best_val_accuracy:.4f} > {val_accuracy}. Saving model...--')
            best_val_accuracy = val_accuracy
            self.save_model()
        
        return best_val_accuracy

    def evaluate(self, X, y, batch_size, regularization, verbose=True):
        n_data = len(X) # length of val images   
        num_batches = (n_data + batch_size - 1) // batch_size 
        total_loss = 0 
        total_correct = 0

        for i in range(0, n_data, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            y_pred  = self.forward(X_batch)

            loss = self.regularized_cross_entropy(self.layers, y_pred, y_batch, regularization, len(X_batch))
            total_loss += loss

            preds = np.argmax(y_pred, axis=1)
            total_correct += np.sum(preds == y_batch)

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