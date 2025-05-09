import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, stride=1, padding=0),  # 降维
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels // 4, kernel_size=3, stride=1, padding=1),  # 3x3 卷积
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1, stride=1, padding=0)  # 升维
        )

    def forward(self, x):
        return x + self.block(x)  # 残差连接

