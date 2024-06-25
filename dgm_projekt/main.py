import model

dirac_gan = model.Model()

parameters, gradients = dirac_gan.generate_trajectory()

print(parameters, gradients)

dirac_gan.train()
