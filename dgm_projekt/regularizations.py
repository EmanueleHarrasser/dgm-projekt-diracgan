import torch.nn as nn
import torch
import torch.nn.utils



def get_regularization(mode, prediction, input, gamma=2):
    if mode in ['GP', 'CRGP']:
        return gradient_penalty(prediction, input, gamma)
    if mode == 'WGP':
        return WGP(prediction, input, gamma)


def gradient_penalty(prediction, input, gamma) :
    gradient = torch.autograd.grad(outputs=prediction, inputs=input, create_graph=True)[0]
    return (gamma/2) * gradient.pow(2) 

def WGP(prediction, input, gamma):
    gradient = torch.autograd.grad(outputs=prediction, inputs=input, create_graph=True)[0].abs()
    return (gamma/2) * (gradient-0.5).pow(2)

#def zero_centered_gradient_penalty(grad):
