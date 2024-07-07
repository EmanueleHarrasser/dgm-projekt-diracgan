import torch.nn as nn
import torch
import torch.nn.utils
import losses
from regularizations import get_regularization

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

class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.ema = None
        self.num_updates = 0

    def update(self, value):
        if self.ema is None:
            self.ema = value.clone()
        else:
            self.ema = self.decay * self.ema + (1.0 - self.decay) * value
        self.num_updates += 1

    def get_ema(self):
        if self.ema is None:
            return None
        debias_factor = 1 - self.decay ** self.num_updates
        return self.ema / debias_factor

    def reset(self):
        self.ema = None
        self.num_updates = 0


class Model(object):
    def __init__(self ,iterations = 1000, learning_rate = .05):

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
        self.moving_average = False
        self.ema_gen = None
        self.ema_real = None

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
    
    def set_regularization_loss(self,regularization_loss):
        self.regularization_loss = regularization_loss
        
        # set gamma
        if regularization_loss in ['CRGP']:
            self.gamma = 1
        elif regularization_loss != '':
            self.gamma = 0.5
        else:
            self.gamma = 0
        
        # Turn on moving average if needed
        if regularization_loss in ['LeCam']:
            self.ema_gen = EMA(0.99)
            self.ema_real = EMA(0.99)
        else:
            self.ema_gen = None
            self.ema_real = None
            
    def set_instance_noise(self,instance_noise):
        self.instance_noise = instance_noise

    def get_vectors(self, steps = 10,size = 2):
        
        real_pred = 0 
        gen_pred = 0 #initialize variables not to break loss function

        vectors = []
        
        generator_parameters = torch.linspace(start=-2, end=2, steps=steps)     #x coordinates
        discriminator_parameters = torch.linspace(start=-2, end=2, steps=steps) #y coordinates

        if self.regularization_loss == 'LeCam':
            self.ema_real.reset()
            self.ema_gen.reset()

        for generator_parameter in generator_parameters:
            for discriminator_parameter in discriminator_parameters:
                #print(generator_parameter,discriminator_parameter)

                self.reset_gradients()

                self.generator.parameter.data = torch.FloatTensor([[generator_parameter]]) 
                self.discriminator.linear.weight.data = torch.FloatTensor([[discriminator_parameter]])

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
                              
                if self.regularization_loss == 'LeCam':
                    self.ema_gen.update(gen_pred)
                    self.ema_real.update(real_pred)
                    ema_fake = self.ema_gen.get_ema().detach()
                    ema_real = self.ema_real.get_ema().detach()
                
                if self.regularization_loss:
                    if self.regularization_loss == 'SimpleLeCam':
                        regularization_term = get_regularization(self.regularization_loss, real_pred, gen_pred, self.gamma)
                    elif self.regularization_loss == 'LeCam':
                        regularization_term = get_regularization(self.regularization_loss, real_pred, gen_pred, self.gamma, ema_fake=ema_fake, ema_real=ema_real)
                    else:
                        regularization_term = get_regularization(self.regularization_loss, real_pred, real_samples, self.gamma) # calculate regularization loss    
                    discriminator_loss += regularization_term   

                discriminator_loss.backward() #backward propagation to get gradients
                

                discriminator_gradient = self.discriminator.linear.weight.grad.item()

                vectors.append((generator_parameter,discriminator_parameter,generator_gradient, discriminator_gradient))
                


        return torch.FloatTensor(vectors)

    def train(self):

        ret1 = []
        ret2 = []
        real_pred = 0
        gen_pred = 0 #initialize variables not to break loss function
        n_d = self.n_d

        self.generator.parameter.data = torch.FloatTensor([[1]]) #set generator weights
        self.discriminator.linear.weight.data = torch.FloatTensor([[1]])  #set discriminator weights

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

                if self.regularization_loss == 'LeCam':
                    self.ema_gen.update(gen_pred)
                    self.ema_real.update(real_pred)
                    ema_fake = self.ema_gen.get_ema().detach()
                    ema_real = self.ema_real.get_ema().detach()
                
                if self.regularization_loss:
                    if self.regularization_loss == 'SimpleLeCam':
                        regularization_term = get_regularization(self.regularization_loss, real_pred, gen_pred, self.gamma)
                    elif self.regularization_loss == 'LeCam':
                        regularization_term = get_regularization(self.regularization_loss, real_pred, gen_pred, self.gamma, ema_fake=ema_fake, ema_real=ema_real)
                        #print(regularization_term)
                    else:
                        regularization_term = get_regularization(self.regularization_loss, real_pred, real_samples, self.gamma) # calculate regularization loss    
                    discriminator_loss += regularization_term
                    
                discriminator_loss.backward()#backward propagate to get gradients

                self.discriminator_optimizer.step()#take a step into gradient direction
                if self.clamp != 0 and self.regularization_loss != 'WGP':
                    self.discriminator.linear.weight.data = self.discriminator.linear.weight.data.clamp_(-self.clamp,self.clamp)

            parameter_history.append((self.generator.parameter.data.item(),
                                      self.discriminator.linear.weight.data.item()))
            ret1.append(self.generator.parameter.data.item())
            ret2.append(self.discriminator.linear.weight.data.item())
        return ret1, ret2

    
    