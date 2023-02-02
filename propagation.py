import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import plotly.express as px
import json
from activation_functions import activation_function

def weight_initialization(n_inputs, layers):
    W = {}
    for i, layer in enumerate(layers):
        if i == 0:
            # pesos[i] = np.random.rand(n_entradas, capa)
            W[i] = np.full((n_inputs, layer), 0.5)
        else:
            # pesos[i] = np.random.rand(capas[i-1], capa)
            W[i] = np.full((layers[i-1], layer), 0.5)
    return W

def propagation_function(W, x, activation, a=1, b=0):
    v_layers = {}
    y_layers = {}

    for i, w_layer in enumerate(W):
        if i == 0:
            v_layer = np.matmul(x, W[w_layer]) # np.dot(j, W[w_layer])
            v_layers[i] = v_layer
            y_layer = activation_function(v_layer, activation, a, b)
            y_layers[i] = y_layer
        else:
            v_layer = np.matmul(y_layers[i-1], W[w_layer])
            v_layers[i] = v_layer
            y_layer = activation_function(v_layer, activation, a, b)
            y_layers[i] = y_layer
    return y_layers, v_layers