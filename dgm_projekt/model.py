import torch.nn as nn
import torch
import torch.nn.utils
import losses
from regularizations import get_regularization
import numpy as np 

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1, bias=False)

    def forward(self, noise):
        return self.linear.forward(noise)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.parameter = nn.Parameter()

    def forward(self):
        return -self.parameter
    


class Model(object):
    def __init__(self, iterations = 1000, learning_rate = 0.05):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator_optimizer = torch.optim.SGD(params= self.generator.parameters(), lr = learning_rate)
        self.discriminator_optimizer = torch.optim.SGD(params= self.discriminator.parameters(), lr = learning_rate)
        self.loss_type = 'GAN'
        self.regularization_loss = ''
        self.iterations = iterations
        self.n_d = 1
        self.clamp = 0
        self.gamma = 0
        self.instance_noise = False

    def reset_gradients(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()
    
    def set_loss(self,loss_type):
        self.loss_type = loss_type
        if loss_type in ['WGAN']:
            self.n_d = 5
            self.clamp = 0.5
        else:
            self.n_d = 1
            self.clamp = 0
    
    def set_regularization_loss(self, regularization_loss):
        self.regularization_loss = regularization_loss
        if regularization_loss in ['GP','WGP']:
            self.gamma = 0.5
        elif regularization_loss in ['CRGP']:
            self.gamma = 1
        else:
            self.gamma = 0
            
    def set_instance_noise(self,instance_noise):
        self.instance_noise = instance_noise

    def get_vectors(self, interval=(-2, 2), steps=10):
        real_pred = 0 
        gen_pred = 0 #initialize variables not to break loss function

        generator_parameters = torch.linspace(start=interval[0], end=interval[1], steps=steps)     #x coordinates
        discriminator_parameters = torch.linspace(start=interval[0], end=interval[1], steps=steps) #y coordinates      

        X, Y = np.meshgrid(generator_parameters, discriminator_parameters) # gen: theta, dis: psi 
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                self.reset_gradients()

                self.generator.parameter.data = torch.FloatTensor([[X[i, j]]]) 
                self.discriminator.linear.weight.data = torch.FloatTensor([[Y[i, j]]])

                gen_pred =  self.discriminator.forward(self.generator.forward()) #forward propagate discriminator with generated data

                generator_loss = losses.get_loss(gen_pred,real_pred,'generator',self.loss_type) #generator loss

                generator_loss.backward() #backward propagation to get gradients
                generator_gradient = self.generator.parameter.grad.item()

                self.reset_gradients()            
                
                real_samples = torch.zeros(1)# create samples (so we can use them for the regularization loss)
                
                real_samples.requires_grad = True 

                real_pred = self.discriminator.forward(torch.FloatTensor(real_samples)) #forward propagate discriminator with real function output

                gen_pred= self.discriminator.forward(self.generator.forward()) #forward propagate discriminator with generated data

                discriminator_loss =  losses.get_loss(gen_pred,real_pred,'discriminator',self.loss_type) #discriminator loss
                              
                if self.regularization_loss:
                    discriminator_loss += get_regularization(self.regularization_loss, real_pred, real_samples, self.gamma) # calculate regularization loss    

                discriminator_loss.backward() #backward propagation to get gradients

                discriminator_gradient = self.discriminator.linear.weight.grad.item()

                U[i, j] = -generator_gradient * self.generator_optimizer.param_groups[-1]["lr"]
                V[i, j] = -discriminator_gradient * self.discriminator_optimizer.param_groups[-1]["lr"]
                
        return X, Y, U, V

    def train(self, init_theta=1, init_psi=1):

        ret1 = []
        ret2 = []
        real_pred = 0
        gen_pred = 0 #initialize variables not to break loss function
        n_d = self.n_d

        self.generator.parameter.data = torch.FloatTensor([[init_theta]]) #set generator weights
        self.discriminator.linear.weight.data = torch.FloatTensor([[init_psi]])  #set discriminator weights

        parameter_history = []

        for _ in range(self.iterations):

            for i in range(n_d):
                self.reset_gradients()
                
                gen_pred = self.discriminator.forward(self.generator.forward()) #forward propagate discriminator with generated data

                if i == n_d-1:
                    generator_loss = losses.get_loss(gen_pred,real_pred,'generator',self.loss_type) #compte generator loss

                    generator_loss.backward() #backward propagate to get gradients

                    self.generator_optimizer.step() #walk in gradient direction
        
                    self.generator_optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()#reset gradients

                real_samples = torch.zeros(1)# create samples (so we can use them for the regularization loss)
                
                if self.instance_noise:
                    real_samples += torch.normal(mean=0., std=0.5, size=tuple([1])) # add instance noise

                real_samples.requires_grad = True 
                
                real_pred = self.discriminator.forward(torch.FloatTensor(real_samples)) #forward propagate samples from real function
                gen_pred = self.discriminator.forward(self.generator.forward()) #forward propagate discriminator with generated data
                
                discriminator_loss =  losses.get_loss(gen_pred,real_pred,'discriminator',self.loss_type) #compute discriminator loss
                
                if self.regularization_loss:
                    discriminator_loss += get_regularization(self.regularization_loss, real_pred, real_samples, self.gamma) # calculate regularization loss    
                    
                discriminator_loss.backward()#backward propagate to get gradients

                self.discriminator_optimizer.step()#take a step into gradient direction
                if self.clamp != 0 and self.regularization_loss != 'WGP':
                    self.discriminator.linear.weight.data = self.discriminator.linear.weight.data.clamp_(-self.clamp,self.clamp)

            parameter_history.append((self.generator.parameter.data.item(),
                                      self.discriminator.linear.weight.data.item()))
            ret1.append(self.generator.parameter.data.item())
            ret2.append(self.discriminator.linear.weight.data.item())
        return ret1, ret2

    
    