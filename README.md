# dgm-projekt-diracgan
Implementation of a Dirac GAN

gui.py contains a simple gui
you can combine losses and regularization strategies and get a (non-animated graph) as output

web.py contains a bokeh implementation of animated graphs of:
1: Standard GAN
2: Non-saturating GAN
3: Wasserstein GAN
4: WGAN-GP
5: GAN with Instance Noise
6: GAN with Gradient Penalty
7: Gan with Gradient Penalty and critical gamma

to run web.py execute:
    bokeh serve --show web.py