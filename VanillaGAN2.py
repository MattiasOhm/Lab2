import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from helperfunctions import get_next_run_number

# This file is for both defining the model, training and evaluation.
# It is a simple GAN with the loss function asked for in task 2.2 (Atleast I think that is what was asked for)
# The model is not saved as I think the needed demonstration is sent to Tensorboard for this task.
# For evaluation: 
#   - images are being generated in each epoch and sent to tensorboard, it shows the improvement over time
#   - the losses are being tracked and sent to tensorboard

class Generator(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.layers_gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers_gen(x)
    
class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.layers_disc = nn.Sequential(
        nn.Linear(img_size, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers_disc(x)
    
# Hyperparameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 3e-4
# Iterations seems to be number of batches, 60.000 size dataset -> 32 batchsize -> 1 epoch ~ 1800 iterations
epochs = 55 
batch_size = 32
z_dim = 100
img_size = 784

# Getting the data
transform = transforms.ToTensor()

dataset = datasets.MNIST(root='data/', download=True, transform=transform)
loader = DataLoader(dataset, batch_size)

# Setup for evaluation

constant_noise = torch.randn(batch_size, z_dim).to(device)
base_dir = "runs/VanillaGANv2"
next_number = get_next_run_number(base_dir)
writer_fake = SummaryWriter(f'runs/VanillaGANv2/fake{next_number}')
writer_real = SummaryWriter(f'runs/VanillaGANv2/real{next_number}')
writer_loss = SummaryWriter(f'runs/VanillaGANv2/loss{next_number}')

# Setup for training

gen = Generator(z_dim, img_size).to(device)
disc = Discriminator(img_size).to(device)
optimizer_gen = torch.optim.Adam(gen.parameters(), lr)
optimizer_disc = torch.optim.Adam(disc.parameters(), lr)
eps = 1e-8 # To prevent taking log of 0 in the loss function
iterations = 0
total_disc_loss = 0.0
total_gen_loss = 0.0

# Training loop

for epoch in range(epochs):
    for batch, (real_images, _) in enumerate(loader): # We don't actually use the real labels, we just give a real image label =1, and fake image label =0
        iterations += 1
        # Discriminator:
        real_images = real_images.flatten(start_dim=1).to(device)
        batch_size = real_images.size(0)
        D_real = disc(real_images)
        D_loss_real = -torch.mean(torch.log(D_real + eps))
        noise=torch.randn((batch_size, z_dim)).to(device)
        fake_images = gen(noise)
        D_fake = disc(fake_images.detach())
        D_loss_fake = -torch.mean(torch.log(1.0 - D_fake + eps))
        D_loss = (D_loss_real + D_loss_fake) / 2
        disc.zero_grad()
        D_loss.backward()
        optimizer_disc.step()
       
        # Generator:
        D_fake = disc(fake_images)
        G_loss = -torch.mean(torch.log(D_fake + eps))
        gen.zero_grad()
        G_loss.backward()
        optimizer_gen.step()

        total_disc_loss += D_loss.item()
        total_gen_loss += G_loss.item()

        # Evaluation
        if batch == 0:
            print(f'Epoch: {epoch}  | Discriminator loss: {D_loss:.3f} | Generator loss: {G_loss:.3f} | Iterations: {iterations}')
        
            with torch.no_grad():
                fake_images = gen(constant_noise).reshape(-1, 1, 28, 28)
                fake_image_grid = make_grid(fake_images)
                writer_fake.add_image("Fake images", fake_image_grid, global_step=epoch)
                real_images = real_images.reshape(-1, 1, 28, 28)
                real_image_grid = make_grid(real_images)
                writer_real.add_image("Real images", real_image_grid, global_step=epoch)

    avg_disc_loss = total_disc_loss / len(loader)
    avg_gen_loss = total_gen_loss / len(loader)
    summed_losses = avg_gen_loss + avg_disc_loss
    writer_loss.add_scalar("Average Discriminator Loss", avg_disc_loss, global_step=epoch)
    writer_loss.add_scalar("Average Generator Loss", avg_gen_loss, global_step=epoch)
    writer_loss.add_scalar("Summed loss", summed_losses, global_step=epoch)
    total_disc_loss = 0.0
    total_gen_loss = 0.0