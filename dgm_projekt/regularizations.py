import torch.nn as nn
import torch
import torch.nn.utils



def get_regularization(mode, prediction, input, gamma=2, ema_real=None, ema_fake=None):
    if mode in ['GP', 'CRGP']:
        return gradient_penalty(prediction, input, gamma)
    if mode == 'WGP':
        return WGP(prediction, input, gamma)
    if mode == 'DRAGAN':
        return DRAGAN(prediction, input, gamma)
    if mode == 'SimpleLeCam':
        return SLeCam(prediction, input, gamma)
    if mode == 'LeCam':
        return lecamreg(prediction, input, ema_real, ema_fake, gamma)


def gradient_penalty(prediction, input, gamma) :
    gradient = torch.autograd.grad(outputs=prediction, inputs=input, create_graph=True)[0]
    return (gamma/2) * gradient.pow(2) 

def WGP(prediction, input, gamma):
    gradient = torch.autograd.grad(outputs=prediction, inputs=input, create_graph=True)[0].abs()
    return (gamma/2) * (gradient-0.5).pow(2)

def DRAGAN(prediction, input, gamma):
    gradient = torch.autograd.grad(outputs=prediction, inputs=input, create_graph=True)[0].abs()
    return (gamma/2) * (gradient-1)**2

def SLeCam(real, fake, gamma):
    return (gamma/2) * torch.abs(real - fake)

def lecamreg(real, fake, ema_real, ema_fake, gamma):
    D_lecam_real = torch.square(real - ema_fake)
    D_lecam_fake = torch.square(ema_real - fake)
    D_reg = gamma * (D_lecam_real + D_lecam_fake)
    return D_reg