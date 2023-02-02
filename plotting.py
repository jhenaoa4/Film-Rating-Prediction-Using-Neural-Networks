import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import plotly.express as px
import json
import plotly.express as px
import scipy.io

def plot_gradients(layers, local_gradient, eta, path):
    '''Grafica el seguimiento de gradientes'''

    for i in range(len(layers)-1,-1,-1):
        plt.plot(range(len(local_gradient[i])),local_gradient[i])
    plt.title('Local gradient, layers:'+str(layers)+', eta='+str(eta))
    plt.xlabel('Epoch')
    plt.ylabel('Local gradient')
    plt.legend(range(len(layers)-1,-1,-1))
    plt.savefig(path+"figures\\local_gradient"+str(layers)+'eta'+str(eta)+".png")
    plt.show()

def plot_errors(errors, layers, eta, path, test, classes=[0,1]):
    '''Grafica el seguimiento de errores'''
    for i in classes:
        plt.plot(range(len(errors[i])), errors[i])
    plt.title('Error, layers:'+str(layers)+', eta='+str(eta)+', '+test)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend(classes)
    plt.savefig(path+"figures\\error"+str(layers)+'eta'+str(eta)+test+".png")
    plt.show()

def plot_energy(energies, layers, eta, path, test):
    '''Grafica el seguimiento de energía'''
    plt.plot(range(len(energies)),energies)
    plt.title('Mean energy, layers:'+str(layers)+', eta='+str(eta)+', '+test)
    plt.xlabel('Epoch')
    plt.ylabel('Mean energy')
    plt.savefig(path+"figures\\energy"+str(layers)+'eta'+str(eta)+test+".png")
    plt.show()
