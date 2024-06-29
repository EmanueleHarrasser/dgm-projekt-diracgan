import torch.nn as nn
import torch
import torch.nn.utils
import losses

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1, bias=False)
        self.linear.weight.data.fill_(1)

    def forward(self, noise):
        return self.linear.forward(noise)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.parameter = nn.Parameter(torch.ones(1, 1))

    def forward(self,repeats):
        return torch.repeat_interleave(self.parameter, repeats= repeats ,dim=0)
    


class Model(object):
    def __init__(self,batch_size = 1 ,iterations = 1000, learning_rate = .05):

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator_optimizer = torch.optim.SGD(params= self.generator.parameters(), lr = learning_rate,momentum=0)
        self.discriminator_optimizer = torch.optim.SGD(params= self.discriminator.parameters(), lr = learning_rate,momentum=0)
        self.loss_type = 'GAN'
        self.regularization_loss = None
        self.batch_size = batch_size
        self.iterations = iterations

    def set_loss(self,loss_type):
        self.loss_type = loss_type

    def get_vectors(self, steps = 10,size = 2):
        
        real_prediction = 0 
        fake_prediction = 0 #initialize variables not to break loss function

        gradients = []
        locations = []
        
        generator_parameters = torch.linspace(start=-2, end=2, steps=steps)     #x coordinates
        discriminator_parameters = torch.linspace(start=-2, end=2, steps=steps) #y coordinates

        

        for generator_parameter in generator_parameters:
            for discriminator_parameter in discriminator_parameters:
                #print(generator_parameter,discriminator_parameter)

                locations.append((generator_parameter,discriminator_parameter)) #append starting point of vector

                self.generator.parameter.data = generator_parameter.view(1,1) #set generator weights
                self.discriminator.linear.weight.data = discriminator_parameter.view(1,1) #set discriminator weights


                self.generator_optimizer.zero_grad() 
                self.discriminator_optimizer.zero_grad() #reset gradients

                fake_prediction =  self.discriminator.forward(self.generator.forward(1)) #forward propagate discriminator with generated data

                generator_loss = losses.get_loss(fake_prediction,real_prediction,'generator',self.loss_type) #generator loss

                generator_loss.backward() #backward propagation to get gradients
                generator_gradient = self.generator.parameter.grad.item()

                self.generator_optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()#reset gradients

                real_prediction = self.discriminator.forward(torch.FloatTensor([[0] for i in range(self.batch_size)])) #forward propagate discriminator with real function output

                fake_prediction = self.discriminator.forward(self.generator(1)) #forward propagate discriminator with generated data

                discriminator_loss =  losses.get_loss(fake_prediction,real_prediction,'discriminator',self.loss_type) #discriminator loss

                discriminator_loss.backward() #backward propagation to get gradients

                discriminator_gradient = self.discriminator.linear.weight.grad.item()

                gradients.append((generator_gradient, discriminator_gradient))


        return torch.FloatTensor(locations), torch.FloatTensor(gradients)

    def train(self):
        """
        Method trains the DiracGAN
        :param instance_noise: (bool) If true instance noise is utilized
        :param (torch.Tensor) History of generator and discriminator parameters [training iterations, 2 (gen., dis.)]
        """
        real_prediction = 0
        fake_prediction = 0 #initialize variables not to break loss function

        self.generator.parameter.data = torch.FloatTensor([[1]]) #set generator weights
        self.discriminator.linear.weight.data = torch.FloatTensor([[1]])  #set discriminator weights

        parameter_history = []

        for _ in range(self.iterations):

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad() #reset gradients
 
            fake_prediction = self.discriminator.forward(self.generator.forward(  self.batch_size )) #forward propagate discriminator with generated data

            generator_loss = losses.get_loss(fake_prediction,real_prediction,'generator',self.loss_type) #compte generator loss

            generator_loss.backward() #backward propagate to get gradients

            self.generator_optimizer.step() #walk in gradient direction
 
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()#reset gradietns

            real_prediction = self.discriminator.forward(torch.FloatTensor([[0]for i in range(self.batch_size)])) #forward propagate samples from real function

            fake_prediction = self.discriminator.forward(self.generator.forward(self.batch_size)) #forward propagate discriminator with generated data

            discriminator_loss =  losses.get_loss(fake_prediction,real_prediction,'discriminator',self.loss_type) #compute discriminator loss

            discriminator_loss.backward()#backward propagate to get gradients

            self.discriminator_optimizer.step()#take a step into gradient direction

            parameter_history.append((self.generator.parameter.data.item(),
                                      self.discriminator.linear.weight.data.item()))
            
        return torch.tensor(parameter_history)

    
    