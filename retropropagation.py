import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import plotly.express as px
import json
# from activation_functions import activation_function      
from activation_functions import phi_derivate

def backpropagation(W, x, y, e, v_layers, y_layers, eta, activation, local_gradient, a=1, b=0):

    '''Gradient of last layer'''
    last_layer = len(W) - 1

    local_grad = np.multiply(e, phi_derivate(v_layers[last_layer], activation, a, b))
    local_gradient[last_layer].append(np.mean(local_grad))
    Delta_w = - eta * np.dot(local_grad.T, y_layers[last_layer-1])
    W[last_layer] = W[last_layer] + Delta_w.T

    '''Gradient of hidden layers'''
    # Opción 1
    for i in range(last_layer-1, -1, -1):
        local_grad = np.multiply(np.dot(local_grad, W[i+1].T), phi_derivate(v_layers[i], activation, a, b))
        local_gradient[i].append(np.mean(local_grad))
        if i == 0:
            Delta_w = - eta * np.dot(x.T, local_grad)
        else:
            Delta_w = - eta * np.dot(y_layers[i-1].T, local_grad)

        W[i] = W[i] + Delta_w

    return W, local_gradient
