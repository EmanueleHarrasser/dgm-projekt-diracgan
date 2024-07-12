# Dirac GAN Implementation

Implementation of a Dirac GAN for the Deep Generative Models course at the TU Darmstadt.

## Overview

**`gui.py`** contains a simple GUI you can combine losses and regularization strategies and get a (non-animated graph) as output.

**`web.py`** contains a Bokeh implementation of animated graphs of:
1. Standard GAN
2. Non-saturating GAN
3. Wasserstein GAN
4. WGAN-GP
5. GAN with Instance Noise
6. GAN with Gradient Penalty
7. GAN with Gradient Penalty and critical gamma

## Usage

To run `web.py`, execute:

```bash
python web.py
```