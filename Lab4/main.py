import numpy as np
from tabulate import tabulate

class NeuralNetwork():
    def __init__(self, input_size:int, output_size:int, hidden_layers:list[int]=[5]):
        self.layers = [np.zeros(input_size)]
        for l in hidden_layers:
            self.layers.append(np.zeros(l))
        self.layers.append(np.zeros(output_size))
        self.len = len(self.layers)
        # the first index is the index of previous layer neuron, the second - of the next layer neuron
        self.weights = self.generate_weights()
        self.biases = self.generate_biases()

    def generate_weights(self):
        layers_weights = []
        for i in range(1, len(self.layers)):
            lw = np.random.uniform(-0.5, 0.5, size=(len(self.layers[i-1]), len(self.layers[i])))
            layers_weights.append(lw)
        return layers_weights
    
    def generate_biases(self):
        biases = self.layers.copy()
        for i in range(self.len):
            biases[i] = np.random.uniform(-0.1, 0.1, size=(biases[i].shape))
        return biases

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def activate_derivative(self, x):
        return x * (1 - x)

    def train(self, x_train, y_train, iterations):
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        train_size = len(y_train)
        for _ in range(iterations):
            for l in range(train_size):
                self.predict(x_train[l])
                vert_errors = self.layers.copy()
                for i in reversed(range(1, self.len)):
                    if i == self.len-1:
                        vert_errors[i] = self.activate_derivative(self.layers[i])*(self.layers[i]-y_train[l])
                    else:
                        vert_errors[i] = self.activate_derivative(self.layers[i])*(np.sum(vert_errors[i+1]*self.weights[i], axis=1))
                    for p in range(len(self.weights[i-1])):
                        for n in range(len(self.weights[i-1][p])):
                            self.weights[i-1][p][n] -= 0.8 * vert_errors[i][n] * self.layers[i-1][p]
                        self.biases[i][n] -= 0.8*vert_errors[i][n]


    def predict(self, inputs):
        self.layers[0] = np.array(inputs)
        for layer in range(1, self.len):
            for n in range(len(self.layers[layer])):
                self.layers[layer][n] = 0
                for p in range(len(self.layers[layer-1])):
                    self.layers[layer][n] += self.layers[layer-1][p]*self.weights[layer-1][p][n]
                self.layers[layer][n] += self.biases[layer][n]
                self.layers[layer][n] = self.activate(self.layers[layer][n])
        return np.array(self.layers[-1])


inputs = 2
outputs = 1
training_size = 1000
iterations = 20
mp = 1000

def dT(dP, dV, P0, V0, T0):
    return (P0*dV + V0*dP + dV*dP)*T0/(P0*V0)



nn = NeuralNetwork(2, 1, [2, 2])
for i in range(3):
    x_train = np.random.randint(1, 25, size=(training_size, inputs))
    y_train = np.array([[dT(dP, dV, 100, 15, 280)/mp] for dP, dV in
                            [x_train[i] for i in range(training_size)]])
    msqw = np.mean(np.array([nn.predict(x) for x in x_train]) - y_train)**2*mp
    print(f'Meansqw of generation {i} = {msqw}')
    nn.train(x_train, y_train, iterations)


print(tabulate([[nn.predict(x_train[i]),
                y_train[i],
                (y_train[i]-nn.predict(x_train[i]))/y_train[i]*100] 
                for i in range(training_size)], 
                headers=['Predicted', 'Expected', 'Dif, %'],
                tablefmt='orgtbl'))

msqw = np.mean(np.array([nn.predict(x) for x in x_train]) - y_train)**2*mp
print(f'Meansqw of result generation = {msqw}')
