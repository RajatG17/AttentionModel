import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image

class RGBDDataset(Dataset):
    def __init__(self,  root_dir, rgb_transform=None, depth_transform=None):
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.rgb_images = []
        self.depth_images = []

        rgb_dir= os.path.join(root_dir, 'rgb')
        depth_dir = os.path.join(root_dir, 'depth')

        for filename in os.listdir(rgb_dir):
            if filename.endswith('.jpg') or filename.endswith('jpeg'):
                rgb_path = os.path.join(rgb_dir, filename)
                depth_path = os.path.join(depth_dir, filename[:-4]+'-depth_raw.png')

                self.rgb_images.append(rgb_path)
                self.depth_images.append(depth_path)

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_image = self.rgb_images[idx]
        depth_image = self.depth_images[idx]

        rgb_image = Image.open(rgb_image).convert('RGB')
        depth_image = Image.open(depth_image).convert('L')

        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        if self.depth_transform:
            depth_image = self.depth_transform(depth_image)

        return rgb_image, depth_image


def get_data_loaders(root_dir, batch_size):
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    depth_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize((0.5), (0.5))
    ])

    dataset = RGBDDataset(root_dir, rgb_transform=rgb_transform, depth_transform=depth_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader