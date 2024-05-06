import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from helperfunctions import get_next_run_number

class Generator(nn.Module):
    def __init__(self, z_dim, img_size, num_classes):
        super().__init__()
        self.layers_gen = nn.Sequential(
            nn.Linear(z_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, img_size),
            nn.Sigmoid()
        )
        self.label_embedding = nn.Embedding(num_classes, num_classes) # Addition for Conditional GAN

    def forward(self, x, labels):
        z = torch.cat((self.label_embedding(labels), x), -1) # Addition for Conditional GAN
        x = self.layers_gen(z)
        x = x.view(x.size(0), 1, 28, 28)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        self.layers_disc = nn.Sequential(
        nn.Linear(img_size + num_classes, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()
        )
        self.label_embedding = nn.Embedding(num_classes, num_classes) # Addition for Conditional GAN
    def forward(self, x, labels):
        x = torch.cat((x.view(x.size(0), -1), self.label_embedding(labels)), -1) # Addition for Conditional GAN
        return self.layers_disc(x)


if __name__ == "__main__":
    # Hyperparameters

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 3e-4
    # Iterations seems to be number of batches, 60.000 size dataset -> 32 batchsize -> 1 epoch ~ 1800 iterations
    epochs = 60
    batch_size = 32
    z_dim = 100
    img_size = 784
    num_classes = 10 # Addition for Conditional GAN

    # Getting the data
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root='data/', download=True, transform=transform)
    loader = DataLoader(dataset, batch_size)

    # Setup for evaluation

    constant_noise = torch.randn(batch_size, z_dim).to(device)
    constant_label = torch.randint(0, 10, (batch_size, )).to(device) # Addidtion for Condtional GAN
    base_dir = "runs/ConditionalGAN"
    next_number = get_next_run_number(base_dir)
    writer_fake = SummaryWriter(f'runs/ConditionalGAN/fake{next_number}')
    writer_real = SummaryWriter(f'runs/ConditionalGAN/real{next_number}')

    # Setup for training

    gen = Generator(z_dim, img_size, num_classes).to(device)
    disc = Discriminator(img_size, num_classes).to(device)
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr)
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr)
    criterion = nn.BCELoss()
    step = 0 # Needed for the tensorboard

    # Training loop

    for epoch in range(epochs):
        for batch, (real_images, labels) in enumerate(loader): # Unlike the previous case, now we need to use labels
            ## Training the discriminator
            # Discriminator on real images:
            real_images = real_images.flatten(start_dim=1).to(device)
            real_labels = labels.to(device) # Addition for Conditional GAN
            D_real = disc(real_images, real_labels)
            D_loss_real = criterion(D_real, torch.ones_like(D_real))

            # Discriminator on fake images:
            noise=torch.randn((batch_size, z_dim)).to(device)
            fake_labels = torch.randint(0, 10, (real_images.size(0), )).to(device)
            fake_images = gen(noise, fake_labels)
            D_fake = disc(fake_images.detach(), fake_labels)
            D_loss_fake = criterion(D_fake, torch.zeros_like(D_fake))

            D_loss = (D_loss_real + D_loss_fake) / 2
            disc.zero_grad()
            D_loss.backward()
            optimizer_disc.step()
        
            ## Training the generator
            G_fake = disc(fake_images, fake_labels)
            G_loss = criterion(G_fake, torch.ones_like(G_fake))
            gen.zero_grad()
            G_loss.backward()
            optimizer_gen.step()

            # Evaluation
            if batch == 0:
                print(f'Epoch: {epoch} / {epochs}  | Discriminator loss: {G_loss:.3f} | Generator loss: {D_loss:.3f}')
            
                with torch.no_grad():
                    fake_images = gen(constant_noise, constant_label).reshape(-1, 1, 28, 28)
                    fake_image_grid = make_grid(fake_images)
                    writer_fake.add_image("Fake images", fake_image_grid, global_step=step)
                    real_images = real_images.reshape(-1, 1, 28, 28)
                    real_image_grid = make_grid(real_images)
                    writer_real.add_image("Real images", real_image_grid, global_step=step)
                    step += 1

    torch.save(gen.state_dict(), 'ConditionalGenerator.pth')