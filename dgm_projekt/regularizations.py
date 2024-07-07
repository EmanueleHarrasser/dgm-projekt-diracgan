import torch.nn as nn
import torch
import torch.nn.utils


def get_regularization(mode, prediction, input, gamma=2 ,discriminator_loss =0,generator_parameter =0,discriminator_parameter = 0):
    if mode in ['GP', 'CRGP']:
        return gradient_penalty(prediction, input, gamma)
    if mode == 'WGP':
        return WGP(prediction, input, gamma)
    if mode == 'CO':
        return consensus_optimization(discriminator_loss,generator_parameter,discriminator_parameter,gamma)


def gradient_penalty(prediction, input, gamma) :
    gradient = torch.autograd.grad(outputs=prediction, inputs=input, create_graph=True)[0]
    return (gamma/2) * gradient.pow(2) 

def WGP(prediction, input, gamma):
    gradient = torch.autograd.grad(outputs=prediction, inputs=input, create_graph=True)[0].abs()
    return (gamma/2) * (gradient-0.5).pow(2)

#def zero_centered_gradient_penalty(grad):

def consensus_optimization(discriminator_loss,generator_parameter,discriminator_parameter,gamma):
    gradient_discriminator = torch.autograd.grad(outputs=discriminator_loss,inputs = discriminator_parameter,create_graph=True)[0]
    gradient_generator = torch.autograd.grad(outputs=discriminator_loss,inputs=generator_parameter,create_graph=True)[0]
    return (gamma/2) * (gradient_discriminator.pow(2) + gradient_generator.pow(2))

