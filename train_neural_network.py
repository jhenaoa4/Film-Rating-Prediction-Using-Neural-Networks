import numpy as np
import pandas as pd
import plotly.express as px
from activation_functions import activation_function
from activation_functions import phi_derivate
from propagation import weight_initialization
from propagation import propagation_function
from errors_and_energy import calculate_error, calculate_energy, calculate_mean_energy, calculate_loss
from retropropagation import backpropagation

#!/usr/bin/python

def network_training(x, y, n_inputs, layers, eta, activation, max_epocas, N_patrones, tol, a=1, b=0, classes = [0,1]):
    '''
    This method trains the network.
    '''
    W = weight_initialization(n_inputs, layers)
    # print(W)

    local_gradient = {}
    for i in range(len(W)):
        local_gradient[i] = []
    
    errors = {}
    for i in classes:
        errors[i] = []

    energies = []

    epoca = 0

    # First Iteration
    y_layers, v_layers = propagation_function(W, x, activation, a, b)
    e = calculate_error(y, y_layers[len(y_layers)-1])
    E = calculate_energy(e)
    E_mean = calculate_mean_energy(E, N_patrones)
    loss = calculate_loss(y, y_layers[len(y_layers)-1])
    # errors.append(np.mean(e))
    for i in classes:
        errors[i].append(loss[i])

    energies.append(E_mean)

    W, local_gradient = backpropagation(W, x, y, e, v_layers, y_layers, eta, activation, local_gradient, a, b)
    epoca += 1

    # Second Iteration
    y_layers, v_layers = propagation_function(W, x, activation, a, b)
    e = calculate_error(y, y_layers[len(y_layers)-1])
    E = calculate_energy(e)
    E_mean = calculate_mean_energy(E, N_patrones)
    loss = calculate_loss(y, y_layers[len(y_layers)-1])
    # errors.append(np.mean(e))
    for i in classes:
        errors[i].append(loss[i])

    energies.append(E_mean)

    W, local_gradient = backpropagation(W, x, y, e, v_layers, y_layers, eta, activation, local_gradient, a, b)
    epoca += 1

    while (epoca < max_epocas and errors[0][-1] > tol and errors[1][-1] > tol and np.abs(local_gradient[len(W)-1][-1]) > 0.0001 and np.abs(local_gradient[len(W)-1][-1] - local_gradient[len(W)-1][-2]) > 0.0001):
    # for i in range(max_epocas):
        y_layers, v_layers = propagation_function(W, x, activation, a, b)
        e = calculate_error(y, y_layers[len(y_layers)-1])
        E = calculate_energy(e)
        E_mean = calculate_mean_energy(E, N_patrones)
        for i in classes:
            errors[i].append(loss[i])

        energies.append(E_mean)

        W, local_gradient = backpropagation(W, x, y, e, v_layers, y_layers, eta, activation, local_gradient, a, b)
        epoca += 1
    

    return local_gradient, y_layers, errors, W, energies