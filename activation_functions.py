import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import plotly.express as px
import json

def activation_function(v, activation, a=1, b=0):
    if activation == 'sigmoid':
        phi = 1 / (1 + np.exp(-v))
    elif activation == 'lineal':
        phi = a*v + b
    elif activation == 'tanh':
        phi = np.tanh(v)
    elif activation == 'softmax':
        phi = np.exp(v) / np.sum(np.exp(v))
    return phi

def phi_derivate(v, activation, a=1, b=0):
    if activation == 'sigmoid':
        phi = 1 / (1 + np.exp(-v))
        phi_derivate = np.multiply(phi,(1 - phi))
    elif activation == 'lineal':
        phi_derivate = [a]*len(v)
    elif activation == 'tanh':
        phi_derivate = 1 - np.power(v, 2)
    elif activation == 'softmax':
        # phi = np.exp(v) / np.sum(np.exp(v))
        # phi_derivate = np.zeros((phi.shape[0], phi.shape[0]))
        # for i in range(phi.shape[0]):
        #     for j in range(phi.shape[0]):
        #         if i == j:
        #             phi_derivate[i][j] = np.multiply(phi[i],(1-phi[i]))
        #         else:
        #             phi_derivate[i][j] = np.multiply(-phi[i]*phi[j])

        # phi_derivate = np.diagflat(v) - np.dot(v, v.T)
        phi = np.exp(v) / np.sum(np.exp(v))
        phi_derivate = np.multiply(phi, (1-phi))

    return phi_derivate