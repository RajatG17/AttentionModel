import torch
import torch.nn as nn
from models.attention_module import SelfAttention

class Discriminator(nn.Module):
    def __init__(self, attention=True):
        super().__init__()
        self.attention = attention

        self.model = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(128) if attention else nn.Identity(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2s(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Flatten(),
            nn.Sigmoid(),
        )