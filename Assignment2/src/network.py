import numpy as np 


class FCNN:
    def __init__(self, layer_sizes,output_activation = "logistic"):
        """
        layer_sizes: e.g., [2, 16, 3] for classification or [1, 10, 1] for regression.
        output_activation: 'logistic' for classification, 'linear' for regression.
        """
        self.layer_sizes = layer_sizes
        self.output_activation = output_activation
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            n = layer_sizes[i] # number of nodes in current layer
            next_n = layer_sizes[i + 1]
            
            # Draw from unit normal and scale by 1/sqrt(n)
            w = np.random.randn(n, next_n) * (1.0 / np.sqrt(n))
            b = np.zeros((1, next_n))
            
            self.weights.append(w)
            self.biases.append(b)

    
    def logistic(self, z):
        return 1 / (1 + np.exp(-np.clip(z,-500,500)))

    def logistic_deriv(self, a):
        return a * (1.0 - a)

    def forward(self, X):
        """
        X: shape (batch_size, input_dim)
        Returns output activations of shape (batch_size, output_dim).
        """

        self.activations = [X]
        self.z_values = []

        current_input = X

        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]

            z = np.dot(current_input,w) + b 
            self.z_values.append(z)
            
            if i < len(self.weights) - 1:
                current_input = self.logistic(z)
            else:
                if self.output_activation == "logistic":
                    current_input = self.logistic(z)
                elif self.output_activation == "linear":
                    current_input = z
                else:
                    raise ValueError(f"Unknown activation")

            self.activations.append(current_input)

        
        return current_input

    def backward(self, y):
        """
        Computes gradients for a single sample or mini-batch.
        y: target shape (batch_size, output_dim)
        """
        N = len(y)

        self.weight_grads = [None] * len(self.weights)
        self.bias_grads = [None] * len(self.biases)


        output = self.activations[-1]

        if self.output_activation == "logistic":
            delta = (output - y) * self.logistic_deriv(output)
        elif self.output_activation == "linear":
            delta = output - y

        for i in reversed(range(len(self.weights))):
            prev_activation = self.activations[i]

            self.weight_grads[i] =  np.dot(prev_activation.T, delta) * (1.0 / N)
            self.bias_grads[i] = np.sum(delta, axis = 0,keepdims=True) * (1.0 / N)
            
            if i > 0:
                # Backpropagate delta to logistic hidden layer
                delta = np.dot(delta, self.weights[i].T) * prev_activation * (1 - prev_activation)

    
    def update_params(self, lr):
        
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.weight_grads[i]
            self.biases[i] -= lr * self.bias_grads[i]


    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=200, lr=0.01, verbose=True):
        """
        Trains the network using stochastic gradient descent. updates weight sample-by-sample
        """
        
        N = len(X_train)
        train_loss_history = []
        val_loss_history = []
        
        for epoch in range(epochs):
            indices = np.random.permutation(N)

            for idx in indices:
                x_sample = X_train[idx: idx + 1]
                y_sample = y_train[idx: idx + 1]

                self.forward(x_sample)
                self.backward(y_sample)
                self.update_params(lr)


            # Average error over training set
            train_preds = self.forward(X_train)
            train_loss = 0.5 * np.mean(np.sum((y_train - train_preds) ** 2, axis=1))
            train_loss_history.append(train_loss)

            if X_val is not None and y_val is not None:
                val_preds = self.forward(X_val)
                val_loss = 0.5 * np.mean(np.sum((y_val - val_preds) ** 2, axis=1))
                val_loss_history.append(val_loss)

            if verbose and ((epoch + 1) % 10 == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch + 1:4d}/{epochs} - Train Loss: {train_loss:.6f}"
                if X_val is not None:
                    msg += f" | Val Loss: {val_loss:.6f}"
                print(msg)

        return train_loss_history, val_loss_history
        
    def predict(self, X):
        preds = self.forward(X)

        if self.output_activation == "logistic":
            if preds.shape[1] == 1:
                return (preds >= 0.5).astype(int)
            return np.argmax(preds,axis=1)
        return preds

