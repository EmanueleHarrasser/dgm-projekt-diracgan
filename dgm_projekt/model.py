import torch.nn as nn
import torch
import torch.nn.utils

HYPERPARAMETERS = {
    "training_iterations": 500,
    "batch_size": 128,
    "lr": .1,
    "in_scale": 0.6,
    "r1_w": 0.2,
    "r2_w": 0.2,
    "gp_w": 0.25,
    "dra_w": 0.1,
    "rlc_af": 1.,
    "rlc_ar": 1.,
    "rlc_w": 0.15
}

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1, bias=False)
        self.linear.weight.data.fill_(1)

    def forward(self, noise):
        return self.linear(noise)
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.ones(1, 1))

    def forward(self,noise):
        return torch.repeat_interleave(self.parameter, repeats= noise.shape[0],dim=0)
    
class Loss_generator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, discriminator_prediction_fake):
        return - nn.functional.softplus(discriminator_prediction_fake).mean()
    
class Loss_discriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, discriminator_prediction_fake, discriminator_prediction_real):
        return nn.functional.softplus(- discriminator_prediction_real).mean() \
               + nn.functional.softplus(discriminator_prediction_fake).mean()
    
def get_noise(batch_size):
        """
        Generates a noise tensor
        :param batch_size: (int) Batch size to be utilized
        :return: (torch.Tensor) Noise tensor
        """
        # return 4. * torch.rand(batch_size, 1, requires_grad=True) - 1.
        return torch.ones(batch_size, 1)

class Model(object):
    def __init__(self):
        """
        Constructor method
        :param generator: (nn.Module) Generator network
        :param discriminator: (nn.Module) Discriminator network
        :param generator_optimizer: (torch.optim.Optimizer) Generator optimizer
        :param discriminator_optimizer: (torch.optim.Optimizer) Discriminator optimizer
        :param generator_loss_function: (nn.Module) Generator loss function
        :param discriminator_loss_function: (nn.Module) Discriminator loss function
        :param regularization_loss: (Optional[nn.Module]) Regularization loss
        """
        # Save parameters
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator_optimizer = torch.optim.SGD(params= [self.generator.parameter], lr = HYPERPARAMETERS["lr"],momentum=0)
        self.discriminator_optimizer = torch.optim.SGD(params= [self.generator.parameter], lr = HYPERPARAMETERS["lr"],momentum=0)
        self.generator_loss_function = Loss_generator()
        self.discriminator_loss_function = Loss_discriminator()
        self.regularization_loss = None

    
    def generate_trajectory(self, steps = 3):
        """
        Method generates gradient trajectory.
        :param steps: (int) Steps to utilize for each parameter in the range of [-2, 2]
        :param instance_noise: (bool) If true instance noise is utilized
        :return: (Tuple[torch.Tensor, torch.Tensor]) Parameters of the shape [steps^2, 2 (gen. dis.)] and parameter
        gradients of the shape [steps^2, 2 (gen. grad., dis. grad.)]
        """
        # Init list to store gradients
        gradients = []
        # Make parameters
        generator_parameters = torch.linspace(start=-2, end=2, steps=steps)
        discriminator_parameters = torch.linspace(start=-2, end=2, steps=steps)

        # Make parameter grid
        generator_parameters, discriminator_parameters = torch.meshgrid(generator_parameters, discriminator_parameters)
        generator_parameters = generator_parameters.reshape(-1)
        discriminator_parameters = discriminator_parameters.reshape(-1)

        # Iterate over parameter combinations
        for generator_parameter, discriminator_parameter in zip(generator_parameters, discriminator_parameters):

            # Set parameters
            self.generator.parameter.data = generator_parameter.view(1,1)
            self.discriminator.linear.weight.data = discriminator_parameter.view(1,1)

            ########## Generator gradient ###########
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Make fake prediction
            fake_prediction =  self.discriminator(self.generator( get_noise( 1 ) ))
            # Compute generator loss
            generator_loss = self.generator_loss_function(fake_prediction)
            # Compute gradients
            generator_loss.backward()
            # Save generator gradient
            generator_gradient = self.generator.parameter.grad.item()
            ########## Discriminator gradient ###########
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Make real prediction
            real_samples = torch.zeros(HYPERPARAMETERS["batch_size"], 1)
            
            real_samples.requires_grad = True
            real_prediction = self.discriminator(real_samples)
            # Make fake prediction
            noise = get_noise(1)
            fake = self.generator(noise)
            fake_prediction = self.discriminator(fake)
            # Compute gradient penalty if utilized
            
            discriminator_loss = torch.zeros(1)
            # Compute generator loss
            discriminator_loss = discriminator_loss + self.discriminator_loss_function(real_prediction, fake_prediction)
            # Compute gradients
            discriminator_loss.backward()
            # Save generator gradient
            discriminator_gradient: float = self.discriminator.linear.weight.grad.item()
            # Save both gradients
            gradients.append((generator_gradient, discriminator_gradient))
        return torch.stack((generator_parameters, discriminator_parameters), dim=-1), torch.tensor(gradients)

    def train(self):
        """
        Method trains the DiracGAN
        :param instance_noise: (bool) If true instance noise is utilized
        :param (torch.Tensor) History of generator and discriminator parameters [training iterations, 2 (gen., dis.)]
        """
        # Set initial weights
        self.generator.parameter.data = torch.ones(1).view(1,1)
        self.discriminator.linear.weight.data = torch.ones(1).view(1,1)
        # Init list to store the parameter history
        parameter_history = []
        # Perform training
        for iteration in range(HYPERPARAMETERS["training_iterations"]):
            ########## Generator training ###########
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Make fake prediction
            fake_prediction = self.discriminator(self.generator( get_noise( HYPERPARAMETERS["batch_size"] ) ))
            # Compute generator loss
            generator_loss = self.generator_loss_function(fake_prediction)
            # Compute gradients
            generator_loss.backward()
            # Perform optimization
            self.generator_optimizer.step()
            ########## Disciminator training ###########
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Make real prediction
            real_samples = torch.zeros(HYPERPARAMETERS["batch_size"], 1)
 
            real_samples.requires_grad = True
            real_prediction = self.discriminator(real_samples)
            # Make fake prediction
            noise = get_noise(HYPERPARAMETERS["batch_size"])
            fake = self.generator(noise)
            fake_prediction = self.discriminator(fake)
           
            discriminator_loss = torch.zeros(1)
            # Compute generator loss
            discriminator_loss = discriminator_loss + self.discriminator_loss_function(real_prediction, fake_prediction)

            # Compute gradients
            discriminator_loss.backward()
            # Perform optimization
            self.discriminator_optimizer.step()
            # Save parameters
            parameter_history.append((self.generator.parameter.data.item(),
                                      self.discriminator.linear.weight.data.item()))
        return torch.tensor(parameter_history)


    