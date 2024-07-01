import tkinter
from tkinter import *
import model
import plotter
import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Button


# standard gan:
gan = model.Model()

# NGAN:
ngan = model.Model()
ngan.set_loss('NGAN')

# WGAN:
wgan = model.Model()
wgan.set_loss('WGAN')

# WGAN-GP:
wgan_gp = model.Model()
wgan_gp.set_loss('WGAN')
wgan_gp.set_regularization_loss('WGP')

# IGP:
igp = model.Model()
igp.set_instance_noise(True)

# GP:
gp = model.Model()
gp.set_regularization_loss('GP')

# CRGP:
crgp = model.Model()
crgp.set_regularization_loss('CRGP')



training_algorithms = [
    (gan, "Standard GAN"),
    (ngan, "Non-saturating GAN"),
    (wgan, "Wasserstein GAN"),
    (wgan_gp, "Wasserstein GAN with GP"),
    (igp, "Instance Noise GAN"),
    (gp, "GAN with Gradient Penalty"),
    (crgp, "Critically Penalized GAN")
]

quiver_locs_theta = np.linspace(-2, 2, 15)
quiver_locs_psi = np.linspace(-2, 2, 15)

plots = []
sources = {}
for model, name in training_algorithms:
    thetas, psis = model.train()
    X,Y = np.meshgrid(quiver_locs_theta, quiver_locs_psi)
    vectors = model.get_vectors()
    U = vectors[:,:2]
    V =vectors[:,2:3]
    p = figure(title=name, x_axis_label='θ', y_axis_label='Ψ')
    #p.quiver(X, Y, U, V, color="navy", line_width=2)
    source = ColumnDataSource(data=dict(x=[], y=[]))
    p.scatter("x", "y", source=source, color="blue", size=8)
    p.x_range.start = np.min(quiver_locs_theta)
    p.x_range.end = np.max(quiver_locs_theta)
    p.y_range.start = np.min(quiver_locs_psi)
    p.y_range.end = np.max(quiver_locs_psi)
    p.scatter([1.], [1.], color="red", size=8)

    sources[name] = (source, thetas, psis)
    plots.append(p)

def get_update(sources, frames):
    frame = 0
    def update():
        nonlocal frame 
        reset = False 
        if frame >= frames:
            reset = True 
            frame = 0 

        for source, thetas, psis in sources.values():
            if reset:
                source.data = dict(x=[], y=[])
            source.stream(dict(x=[thetas[frame]], y=[psis[frame]]), rollover=frames)
        frame += 1 
    return update 

grid = gridplot(plots, ncols=3, width=300, height=300, toolbar_options=dict(logo=None))

curdoc().add_periodic_callback(get_update(sources, 1000), 30)
curdoc().add_root(grid)
