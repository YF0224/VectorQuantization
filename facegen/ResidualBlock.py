import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, 1, 0),  # 1x1 卷积降维
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels // 4, 3, 1, 1),  # 3x3 卷积
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1, 1, 0)  # 1x1 卷积升维
        )
        
    def forward(self, x):
        return x + self.block(x)  # 残差连接