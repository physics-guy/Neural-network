import numpy as np
import pickle
import warnings
warnings.filterwarnings("error")

class Layer_Dense:
    '''Dense layer object'''
    def __init__(self, n_inputs, n_neurons):
        '''Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.'''
        self.weights = np.random.randn(n_neurons, n_inputs) / np.sqrt(n_inputs)
        self.biases = np.zeros(n_neurons)
    
    def forward(self, inputs):
        '''update layer neurons according to weights and biases'''
        self.outputs = np.dot(inputs, self.weights.T) + self.biases
        return self.outputs
    
class Activation_ReLU:
    '''Activation ReLu object'''
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
        return self.outputs
    
    def derivative(self, inputs):
        derivative = np.zeros(inputs.shape)
        for i, rida in enumerate(inputs):
            for j, element in enumerate(rida):
                if element <= 0:
                    derivative[i, j]  = 0
                else:
                    derivative[i, j] = 1

        return derivative
        
class Activation_Softmax:
    '''Activation Soxtmax object'''
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities
        return self.outputs
    
    def derivative(self, inputs):
        pass
    
class Activation_Sigmoid:
    '''Activation Sigmoid object'''
    def forward(self, inputs):
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs
    
    def derivative(self, inputs):
        f = self.forward(inputs)
        derivative = f * (1 - f)
        return derivative

class Loss:
    '''Loss calculation object'''
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    '''Categorical Cross entropy object'''
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def output_error(self, output_activation, y, activation_func_derivative, weighted_outputs):
        return (output_activation - y)
    
class Loss_CrossEntropy(Loss):
    '''Cross entropy loss object'''
    def forward(self, output_activations, y):
        output_activations = np.clip(output_activations, 1e-7, 1-1e-7)
        loss = -np.sum(np.nan_to_num(np.multiply(y, np.log(output_activations)) + np.multiply((1-y), np.log(1-output_activations))), axis=1)
        return loss

    def output_error(self, output_activation, y, activation_func_derivative, weighted_outputs):
        return (output_activation - y)
    
class Loss_Quadratic(Loss):
    '''Quadratic loss object'''
    def forward(self, y_pred, y_true):
        if len(y_pred.shape) == 1:
            loss = np.sum((y_true - y_pred)**2) / 2*y_pred.size
        elif len(y_pred.shape) == 2:
            if len(y_true.shape) == 1:
                y_true = np.array([[1 if i==j else 0 for j in np.arange(y_pred.shape[1])] for i in y_true])
            n = y_pred.shape[1]
            loss = np.sum((y_true - y_pred)**2) / (2*n)
        return loss
    
    def output_error(self, output_activations, y, activation_func_derivative, weighted_outputs):
        if len(y.shape) == 1:
            y = np.array([[1 if i==j else 0 for j in np.arange(output_activations.shape[0])] for i in y])
        delta = np.multiply((output_activations - y), activation_func_derivative(weighted_outputs))
        return delta
    
class NeuralNetwork:
    '''Overall network class'''
    def __init__(self, n_inputs, hidden_layers, n_outputs, loss_object=Loss_Quadratic, \
                 activation_object=Activation_Sigmoid, activation_output_layer=Activation_Sigmoid):
        
        self.structure = [n_inputs] + hidden_layers + [n_outputs]       #set network structure
        self.layers = []                                                #contains all Layer_Dense objects
        self.loss_object = loss_object()                                #chosen loss calculation method
        self.activations = []                                           #contains corresponding activation objects
        #append objects to list
        for prev, current in zip(self.structure[:-1], self.structure[1:]):
            self.layers.append(Layer_Dense(prev, current))
            self.activations.append(activation_object())
        self.activations[-1] = activation_output_layer()
     
    def forward_propagate(self, inputs):
        self.layers[0].forward(inputs)                      #feed inputs to second layer
        self.activations[0].forward(self.layers[0].outputs) #feed weighted output z to activation func
        
        for i in np.arange(1, len(self.layers)):
            self.layers[i].forward(self.activations[i-1].outputs)   #feed previous layer activations a
            self.activations[i].forward(self.layers[i].outputs)     #feed weighted output z to activation func
            
        return self.activations[-1].outputs     #return output layer activations
    
    def predictions(self, inputs):
        predictions = np.argmax(inputs, axis=1) #prediction has the highest activation
        return predictions
    
    def calculate_accuracy(self, predictions, expected):
        accuracy = np.mean(predictions==expected) #accuracy is the arithmetic mean of number of all correct predictions
        return accuracy
    
    def calculate_loss(self, inputs, y, lmbda=0):
        if len(y.shape) == 1:
            y = np.array([[1 if i==j else 0 for j in np.arange(inputs.shape[1])] for i in y])
        loss = self.loss_object.calculate(inputs, y)
        regularization = 0
        for l in self.layers:
            regularization += np.linalg.norm(l.weights)**2
        loss += 0.5*(lmbda/inputs.shape[0]) * regularization
        return loss
    
    def train_SGD(self, train_data, y, learning_rate, epochs, lmbda=0, mini_batch_size=20, print_data=False):
        #convert y to corresponding neurons
        if len(y.shape) == 1:
            y = np.array([[1 if i==j else 0 for j in np.arange(self.structure[-1])] for i in y])
        samples = train_data.shape[0]
        
        for epoch in np.arange(epochs):
            #shuffle test data
            shuffled_indexes = np.arange(len(train_data))
            np.random.shuffle(shuffled_indexes)
            train_data = train_data[shuffled_indexes]
            y = y[shuffled_indexes]
            #divide into mini batches
            num_mini_batches = int(train_data.shape[0]/mini_batch_size)
            mini_batches = np.array_split(train_data, num_mini_batches, axis=0)
            y_mini_batches = np.array_split(y, num_mini_batches, axis=0)
        
            for y_mini_batch, mini_batch in zip(y_mini_batches, mini_batches):
                #activate train data
                activated_mini_batch = Activation_Sigmoid().forward(mini_batch)
                #feed train data to current network
                outputs = self.forward_propagate(mini_batch)
                
                #nabla_b and nabla_w are gradients for the cost function
                nabla_b = [np.zeros(layer.biases.shape) for layer in self.layers]
                nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers]
                
                #output error \delta^L
                delta = self.loss_object.output_error(outputs, y_mini_batch, \
                        self.activations[-1].derivative, self.layers[-1].outputs)
                nabla_b[-1] = np.sum(delta, axis=0)
                nabla_w[-1] = delta.T @ self.activations[-2].outputs
                
                #backpropagate error
                for l in np.arange(len(self.layers)-2, -1, -1):
                    delta = np.multiply((self.layers[l+1].weights.T @ delta.T).T, \
                            self.activations[l].derivative(self.layers[l].outputs))
                    
                    nabla_b[l] = np.sum(delta, axis=0)
                    nabla_w[l] = delta.T @ (self.activations[l-1].outputs \
                            if l!=0 else activated_mini_batch)
                    
                #gradient descent, update weights and biases
                gradient_step = learning_rate/mini_batch.shape[0]
                for l in np.arange(len(self.layers)-1, -1, -1):
                    self.layers[l].biases += -gradient_step * nabla_b[l]
                    
                    self.layers[l].weights = (1 - learning_rate*(lmbda/samples))*self.layers[l].weights - gradient_step * nabla_w[l]
            
            if print_data:
                debug_outputs = self.forward_propagate(train_data)
                loss = self.calculate_loss(debug_outputs, y, lmbda)
                print(f'Epoch {epoch}: loss {loss:.2f}')
        
        def save_network(self, file):
            with open(file, 'wb') as f:
                data = {'structure': self.structure,
                        'loss': self.loss_object,
                        'activations': self.activations,
                        'layers': self.layers}
                pickle.dump(data, f)
        
        def load_network(self, file):
            with open(file, 'rb') as f:
                data = pickle.load(f)
                self.structure = data['structure']
                self.loss_object = data['loss']
                self.activations = data['activations']
                self.layers = data['layers']