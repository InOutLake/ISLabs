import numpy as np
from math import exp

INPUT_SIZE = 5
OUTPUT_SIZE = 2

x = np.random.uniform(low=0., high=1., size=(INPUT_SIZE,))
w = np.random.uniform(low=0., high=1., size=(OUTPUT_SIZE, INPUT_SIZE))
y = np.ones((OUTPUT_SIZE,))

def sigmoid(x: float):
    return 1/(1-exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def calculate_error(y_true, y_pred):
    return np.mean((y_true - y_pred) **  2)

def calculate_gradient(y_true, y_pred, x, w):
    mse_gradient =  2 * (y_pred - y_true)
    der = sigmoid_derivative(y_pred)
    return mse_gradient * der * x

for i in range(OUTPUT_SIZE):
    y[i] = sigmoid(sum(x*w[i]))

print(y)