import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def init_plot():
    plt.ylabel('generator_parameter')
    plt.xlabel('discriminator_parameter')
    plt.gca().set_aspect('equal')

def plot_vectors(vectors):
    plt.quiver(vectors[:, 0], vectors[:, 1], vectors[:, 2], vectors[:, 3])
    
def plot_points(points):
    plt.scatter(points[:,0],points[:,1])

def show_plot():
    plt.show()
