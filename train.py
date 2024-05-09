import torch
import torch.nn as nn
from models.generator import Generator
from models.discriminator import Discriminator
from models.loss import GANLoss
from utils.data_utils import get_data_loaders

def train(data_loader, generator, discriminator, loss_fn, optimizer_g, optimizer_d, device, num_epochs):
    for epoch in range(num_epochs):
        for rgb_images, depth_images in data_loader:
            rgb_images = rgb_images.to(device)
            depth_images = depth_images.to(device)

            optimizer_d.zero_grad()
            noise = torch.randn(rgb_images, depth_images, noise)
            generated_images = generator(rgb_images, depth_images, noise)
            real_output = discriminator(rgb_images, depth_images, rgb_images)
            fake_output = discriminator(rgb_images, depth_images, generated_images.detach())
            discriminator_loss = loss_fn(real_output, True) + loss_fn(fake_output, False)
            discriminator_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            noise = torch.randn(rgb_images.size(0), 100, 1, 1).to(device)
            generated_images = generator(rgb_images, depth_images, noise)
            fake_output = discriminator(rgb_images, depth_images, generated_images)
            generator_loss = loss_fn(fake_output, True)
            generator_loss.backward()
            optimizer_g.step()


        print(f"Epoch [{epoch+1}/{num_epochs}] Generator Loss: {generator_loss.item()} Discriminator Loss: {discriminator_loss.item()}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = 'data/'
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4

    data_loader = get_data_loaders(root_dir, batch_size)
    generator = Generator(noise_dim=100, attention=True).to(device)
    discriminator = Discriminator(attention=True).to(device)
    loss_fn = GANLoss()
    optimizer_d = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas = (0.5, 0.999))
    optimizer_g = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas = (0.5, 0.999))

    train(data_loader, generator, discriminator, loss_fn, optimizer_g, optimizer_d, device, num_epochs)

