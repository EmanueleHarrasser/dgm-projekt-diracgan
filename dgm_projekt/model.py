import torch.nn as nn
import torch
import torch.nn.utils
import losses

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
        return self.parameter
    


class Model(object):
    def __init__(self,iterations = 1000, learning_rate = .05):

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator_optimizer = torch.optim.SGD(params= self.generator.parameters(), lr = learning_rate)
        self.discriminator_optimizer = torch.optim.SGD(params= self.discriminator.parameters(), lr = learning_rate)
        self.loss_type = 'GAN'
        self.iterations = iterations

    def reset_gradients(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()

    def set_loss(self,loss_type):
        self.loss_type = loss_type

    def get_vectors(self, steps = 10,size = 2):
        
        real_pred = 0 
        gen_pred = 0 #initialize variables not to break loss function

        vectors = []
        
        generator_parameters = torch.linspace(start=-2, end=2, steps=steps)     #x coordinates
        discriminator_parameters = torch.linspace(start=-2, end=2, steps=steps) #y coordinates        

        for generator_parameter in generator_parameters:
            for discriminator_parameter in discriminator_parameters:

                self.reset_gradients()

                self.generator.parameter.data = torch.FloatTensor([[generator_parameter]]) 
                self.discriminator.linear.weight.data = torch.FloatTensor([[discriminator_parameter]]) #reset gradients and initialize generator/discriminator parameters
                
                gen_pred =  self.discriminator.forward(self.generator.forward()) #forward propagate discriminator with generated data

                generator_loss = losses.get_loss(gen_pred,real_pred,'generator',self.loss_type) #generator loss

                generator_loss.backward() #backward propagation to get gradients

                generator_gradient = self.generator.parameter.grad.item()

                self.reset_gradients()

                real_pred = self.discriminator.forward(torch.FloatTensor([[0]])) #forward propagate discriminator with real function output

                gen_pred = self.discriminator.forward(self.generator.forward()) #forward propagate discriminator with generated data

                discriminator_loss =  losses.get_loss(gen_pred,real_pred,'discriminator',self.loss_type) #discriminator loss

                discriminator_loss.backward() #backward propagation to get gradients

                discriminator_gradient = self.discriminator.linear.weight.grad.item()

                vectors.append((generator_parameter,discriminator_parameter,-generator_gradient,-discriminator_gradient))


        return torch.FloatTensor(vectors)

    def train(self):
        """
        Method trains the DiracGAN
        :param instance_noise: (bool) If true instance noise is utilized
        :param (torch.Tensor) History of generator and discriminator parameters [training iterations, 2 (gen., dis.)]
        """
        real_pred = 0
        gen_pred = 0 #initialize variables not to break loss function

        self.generator.parameter.data = torch.FloatTensor([[1]]) #set generator weights
        self.discriminator.linear.weight.data = torch.FloatTensor([[1]])  #set discriminator weights

        trail = []

        for _ in range(self.iterations):

            self.reset_gradients()

            gen_pred = self.discriminator.forward(self.generator.forward()) #forward propagate discriminator with generated data

            generator_loss = losses.get_loss(gen_pred,real_pred,'generator',self.loss_type) #compte generator loss

            generator_loss.backward() #backward propagate to get gradients

            self.generator_optimizer.step() #walk in gradient direction
 
            self.reset_gradients()

            real_pred = self.discriminator.forward(torch.FloatTensor([[0]])) #forward propagate samples from real function

            gen_pred = self.discriminator.forward(self.generator.forward()) #forward propagate discriminator with generated data

            discriminator_loss =  losses.get_loss(gen_pred,real_pred,'discriminator',self.loss_type) #compute discriminator loss

            discriminator_loss.backward()#backward propagate to get gradients

            self.discriminator_optimizer.step()#take a step into gradient direction

            trail.append((self.generator.parameter.data.item(),self.discriminator.linear.weight.data.item()))
            
        return torch.tensor(trail)

    
    