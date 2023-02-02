import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import plotly.express as px


def calculate_error(y, y_pred):
    e = y - y_pred
    return e

def calculate_energy(e):
    E = np.sum(np.power(e,2)) / 2
    return E

def calculate_mean_energy(E, N):
    E_promedio = E / N
    return E_promedio

def calculate_loss(y_real, y_pred):
  classes = list(set(y_real))
  err={}
  for i in classes:
    cont = 0
    for j,real in enumerate(y_real):
      if real == i and y_pred[j] != real: cont +=1
    err[i]= cont/len(y_real)

    return err