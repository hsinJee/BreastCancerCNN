from conv import Convolutional
from dense import Dense
from flatten import Flatten
from batch_norm import BatchNormalization
from pooling import Pooling
import numpy as np
import time
import pickle

class CNN:
    def __init__(self): 
        self.layers = []
        self.training = True
    
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
                                output_size=128,  # You can adjust the number of units
                                useSoftmax=False))

            # Output Layer (Softmax for classification)
            self.add_layer(Dense(name='output', 
                                input_size=128, 
                                output_size=2,  # For binary classification (if there are two classes)
                                useSoftmax=True))
            
    def forward(self, image, training):
        # training bool false or true
        for layer in self.layers:
            image = layer.forward(image, training)
            print(f"{layer.name}")
        return image
    
    def backward(self, dout, learning_rate):
        print("--backwards pass--")
        for layer in reversed(self.layers):
            dout = layer.backward(dout, learning_rate)
            print(f"{layer.name}")     
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

        # do not care about val_loss val_accuracy and lr for now ill implement later
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}
        X_train = dataset['train_images']
        y_train = dataset['train_labels']
        num_benign = np.sum(y_train == 0)
        num_malignant = np.sum(y_train == 1)

        # weighted inputs
        total = num_benign + num_malignant
        weight_benign = total / (2 * num_benign)
        weight_malignant = total / (2 * num_malignant)
        class_weights = np.array([weight_benign, weight_malignant])
        
        n_train = len(X_train)
        
        best_accuracy = 0
        best_val_accuracy = -np.inf # start with very low value

        for epoch in range(num_epochs):
            self.training = True
            epoch_loss, epoch_correct = 0, 0
            num_batches = (n_train + batch_size - 1) // batch_size

            indices = np.arange(n_train)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            initial_time = time.time()

            # for evaluation, only evaluate the last 80% of batches
            threshold_batch = int(num_batches * 0.8)

            for i in range(0, n_train, batch_size): 
                batch_num = i // batch_size + 1

                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                
                # pass batch through forward pass
                y_pred = self.forward(X_batch, training=True)

                # number of correct preds
                batch_preds = np.argmax(y_pred, axis=1)

                if self.dataset_name == 'breakHis':
                    y_true = np.argmax(y_batch, axis=1)
                    batch_correct = np.sum(batch_preds == y_true)
                elif self.dataset_name == 'mnist': 
                    batch_correct = np.sum(batch_preds == y_batch)

                epoch_correct += batch_correct
                batch_accuracy = batch_correct / batch_size

                if y_batch.ndim > 1:  # one-hot encoded
                    y_batch = np.argmax(y_batch, axis=1)
                # backward pass
                gradient = y_pred.copy()
                gradient[np.arange(len(X_batch)), y_batch] -= 1
                gradient /= len(X_batch)

                
                # loss function
                loss = self.regularized_cross_entropy(self.layers, y_pred, y_batch, regularization, len(X_batch), class_weights=class_weights)
                epoch_loss += loss
                
                self.backward(gradient, learning_rate)

                history['loss'].append(loss)
                history['accuracy'].append(batch_accuracy)
                
                # print each first and last batch and every 50 batches
                if batch_num <= 500 or batch_num % 50 == 0 or batch_num == num_batches or batch_num == 1:
                    print(f"Batch {batch_num}/{num_batches}, Loss: {loss:.4f}, Accuracy: {batch_accuracy:.4f}, Time: {(time.time() - initial_time):.2f}s")
                    initial_time = time.time()

                # if batch_accuracy > best_accuracy:
                #     print(f"Batch {batch_num}/{num_batches}, Loss: {loss:.4f}, Accuracy increase {best_accuracy} > {batch_accuracy}")
                #     best_accuracy = batch_accuracy

                # if validate: # only start evaluating after threshold
                #     print("hello")
                #     best_val_accuracy = self.call_evaluate(dataset, history, batch_size, best_val_accuracy, regularization, verbose)
            

            # evaluate after epoch 
            if validate:
                self.training = False
                best_val_accuracy = self.call_evaluate(dataset, history, batch_size, best_val_accuracy, regularization, verbose)

            # average loss for this batch
            epoch_loss /= num_batches
            accuracy = epoch_correct / n_train
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {epoch_loss}, Avg Accuracy: {accuracy}, Best Accuracy: {best_accuracy}")

    def call_evaluate(self, dataset, history, batch_size, best_val_accuracy, regularization, verbose):
        print("--Evaluating--")
        # evaluate after epoch 
        X_val = dataset['validation_images']
        y_val = dataset['validation_labels']

        if y_val.ndim > 1:  # one-hot encoded labels
            y_val = np.argmax(y_val, axis=1) 
       
        indices = np.random.permutation(len(X_val))
        X_val = X_val[indices]
        y_val = y_val[indices]

        val_loss, val_accuracy = self.evaluate(
            X_val,
            y_val,
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

            if y_batch.ndim > 1:  # one-hot encoded labels
                y_batch = np.argmax(y_batch, axis=1)

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
