from tinygrad.nn.optim import Adam
from tinygrad import Tensor, nn 
from extra.datasets import fetch_mnist
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path

class InstanceNoise:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, x):
        noise = Tensor.randn(x.shape) * self.sigma
        return x + noise
        
class Discriminator: 
    def __init__(self, input_dim):
        self.l1 = nn.Linear(input_dim, 1024, bias=True)
        self.l2 = nn.Linear(1024, 512, bias=True)
        self.l3 = nn.Linear(512, 256, bias=True)
        self.out = nn.Linear(256, 1, bias=True) 

    def __call__(self, x: Tensor):
        x = self.l1(x).leakyrelu(0.2).dropout(0.3) 
        x = self.l2(x).leakyrelu(0.2).dropout(0.3) 
        x = self.l3(x).leakyrelu(0.2).dropout(0.3) 
        return self.out(x).sigmoid()#.clip(1e-6, 1 - 1e-6)
    
    def parameters(self): 
        return [self.l1.weight, self.l2.weight, self.l3.weight, self.out.weight]

class Generator:
    def __init__(self, latent_space, output_dim):
        self.l1 = nn.Linear(latent_space, 256, bias=True)
        self.l2 = nn.Linear(256, 512, bias=True) 
        self.l3 = nn.Linear(512, 1024, bias=True)
        self.out = nn.Linear(1024, output_dim, bias=True)

    def __call__(self, x: Tensor): 
        x = self.l1(x).leakyrelu(0.2) 
        x = self.l2(x).leakyrelu(0.2) 
        x = self.l3(x).leakyrelu(0.2) 
        return self.out(x).tanh()#.clip(1e-6, 1 - 1e-6)

    def parameters(self): 
        return [self.l1.weight, self.l2.weight, self.l3.weight, self.out.weight]
     
class DataLoader: 
    def __init__(self, data, batch_size: int, output_dim: int): 
        self.data = data
        self.batch_size = batch_size
        self.batch_count = int(np.ceil(len(self.data) / batch_size)) 
        self.output_dim = output_dim

        self.idx = 0 

    def __iter__(self):
        self.idx = 0 
        return self 

    def __next__(self):
        if self.idx >= self.batch_count: raise StopIteration
        samp = np.random.choice(self.data.shape[0], size=(self.batch_size), replace=False) 
        self.idx += 1  
        return Tensor(self.data[samp], requires_grad=True).reshape(-1, self.output_dim)

def feature_matching_loss(real_features: Tensor, fake_features: Tensor) -> Tensor:
    return ((real_features.mean(0) - fake_features.mean()) ** 2).mean() 

def wasserstein_loss(D_real: Tensor, D_fake: Tensor) -> Tensor:
    return D_real.mean() - D_fake.mean() 

def sample(epoch: int, noise: Tensor, out_dir: Path, gen: Generator, nz: int, sample_count = 100):
    generated_images = (gen(noise).reshape(-1, 28, 28) + 1) / 2

    size = int(np.sqrt(sample_count))
    fig, axs = plt.subplots(size, size, figsize=(size, size))
    cnt = 0
    for i in range(size):
        for j in range(size):
            axs[i, j].imshow(generated_images[cnt].squeeze().numpy(), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(out_dir / f"mnist_{epoch}.png")
    plt.close()

if "__main__" in __name__:
    X_train, _, X_test, _ = fetch_mnist()
    X_train = (X_train.astype(np.float64) / 127.5 - 1.0).reshape(-1, 28, 28)
    #X_train = X_train[0:10000]

    image_shape = int(np.prod(X_train.shape[1:])) 
    batch_size = 512
    epochs = 300 
    nz = 100  # Size of z latent vector (input to generator)
    lr = 0.0002
    k = 1 
    sample_interval = 5 
    sample_count = 100

    discriminator = Discriminator(image_shape) 
    generator = Generator(nz, image_shape) 

    optim_d = Adam(discriminator.parameters(), lr=lr, b1=0.5)
    optim_g = Adam(generator.parameters(), lr=lr, b1=0.5)

    dataloader = DataLoader(X_train, batch_size, image_shape) 
    print("Dataloader batch count:", dataloader.batch_count)

    labels_fake = Tensor(np.zeros(batch_size).reshape(-1, 1))
    labels_real = Tensor(np.ones(batch_size).reshape(-1, 1))
    
    output_dir = Path(".").resolve() / "outputs"
    output_dir.mkdir(exist_ok=True)
    ds_noise = Tensor.randn(sample_count, nz, requires_grad=False)

    with Tensor.train():
        sample(0, ds_noise, output_dir, generator, nz, sample_count=sample_count)
        for epoch in range(1, epochs + 1): 
            losses_g, losses_d = 0.0, 0.0
            for batch_idx, batch_x in enumerate(dataloader): 
                for _ in range(k):
                    # train discriminator 
                    optim_d.zero_grad() 
                    
                    # generate fake imgs 
                    noise = Tensor.randn(batch_size, nz)
                    fake_img = generator(noise) 

                    # calc discriminator output 
                    out_d_real = discriminator(batch_x)
                    out_d_fake = discriminator(fake_img.detach()) 

                    # calc loss 
                    loss_d_fake = out_d_fake.binary_crossentropy(labels_fake)
                    loss_d_real = out_d_real.binary_crossentropy(labels_real)
                    loss_d = loss_d_fake + loss_d_real
                    loss_d.backward()
                    
                    optim_d.step() 

                # train generator 
                optim_g.zero_grad() 

                noise = Tensor.randn(batch_size, nz)
                fake_img = generator(noise)
                d_fake = discriminator(fake_img) 

                # calc loss 
                loss_g = d_fake.binary_crossentropy(labels_real) 
                loss_g.backward()
                
                optim_g.step()
            
                losses_d += loss_d.item()
                losses_g += loss_g.item() 

            if epoch % sample_interval == 0: 
                sample(epoch, ds_noise, output_dir, generator, nz, sample_count=sample_count)

            print(f"Epoch [{epoch}/{epochs}] Loss D: {losses_d/dataloader.batch_count:.4f}, Loss G: {losses_g/dataloader.batch_count:.4f}")
    print("Finished!")