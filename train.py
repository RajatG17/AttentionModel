import argparse

import torch
import torch.nn as nn
from models.generator import Generator
from models.discriminator import Discriminator
from models.loss import GANLoss
from utils.data_utils import get_data_loaders
from generate import generate_images

def train(data_loader, generator, discriminator, loss_fn, optimizer_g, optimizer_d, device, num_epochs):
    for epoch in range(num_epochs):
        for rgb_images, depth_images in data_loader:
            rgb_images = rgb_images.to(device)
            depth_images = depth_images.to(device)

            optimizer_d.zero_grad()
            noise = torch.randn(rgb_images.size(0), 100, 1, 1).to(device)
            generated_images = generator(rgb_images, depth_images, noise).to(device)
            real_output = discriminator(rgb_images, depth_images, rgb_images).to(device)
            fake_output = discriminator(rgb_images, depth_images, generated_images.detach()).to(device)
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
        if (epoch + 1) % 10 == 0:
            generate_images(generator, 25, device)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for optimizers')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use for training')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    device = torch.device(args.device)
    data_loader = get_data_loaders(args.root_dir, args.batch_size)

    # Initialize models
    generator = Generator(noise_dim=100, attention=True).to(device)
    discriminator = Discriminator(attention=True).to(device)

    loss_fn = GANLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    train(data_loader, generator, discriminator, loss_fn, optimizer_g, optimizer_d, device, args.num_epochs)