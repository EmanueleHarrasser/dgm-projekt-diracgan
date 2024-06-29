import torch.nn as nn
import torch
import torch.nn.utils

implemented_losses = ['GAN']

def get_loss(fake,real,mode,type):
    if type == 'GAN':
        return GAN_loss(fake,real,mode)
    if type == 'NGAN':
        return NGAN_loss(fake,real,mode)

def GAN_loss(fake,real,mode):
    if mode == 'discriminator':
        return nn.functional.softplus(- real)  + nn.functional.softplus(fake)
    if mode == 'generator':
        return - nn.functional.softplus(fake)

def NGAN_loss(fake,real,mode):
    if mode == 'discriminator':
        return GAN_loss(fake,real,mode)
    if mode == 'generator':
        return nn.functional.softplus(- fake)

