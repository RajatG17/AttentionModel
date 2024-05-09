import torch
from models.generator import Generator
from utils.visualize import save_image

def generate_images(generator, num_images, device):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, 100, 1, 1).to(device)
        rgb_images = torch.randn(num_images, 3, 64, 64).to(device)
        depth_images = torch.randn(num_images, 1, 64, 64).to(device)
        generated_images = generator(rgb_images, depth_images, noise)
        save_image(generated_images, "generated_images.png")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "lsun_bedroom256.pth"
    num_images = 10

    generator = Generator(noise_dim=100, attention=True).to(device)
    generator.load_state_dict(torch.load(model_path))

    generate_images(generator, num_images, device)