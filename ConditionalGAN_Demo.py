import ConditionalGAN
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

# This file just loads the already trained model and generates 10 images of each number 0-9

z_dim = 100
img_size = 28 * 28
num_classes = 10
model = ConditionalGAN.Generator(z_dim, img_size, num_classes)
model.load_state_dict(torch.load("ConditionalGenerator.pth"))
num_images = 100
constant_noise = torch.randn(num_images, z_dim)
labels = torch.repeat_interleave(torch.arange(0,10), repeats=10)
images = model(constant_noise, labels)
writer_conditional = SummaryWriter(f'runs/ConditionalGAN/generated')
image_grid = make_grid(images, nrow=10)
writer_conditional.add_image("Generated images", image_grid)