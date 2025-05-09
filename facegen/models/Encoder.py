import torch
import torch.nn as nn
from .ResidualBlock import ResidualBlock

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, res_nums):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels//4, 4, 2, 1),  # 256x256
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels//4, hidden_channels//2, 4, 2, 1),  # 128x128
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels//2, hidden_channels, 4, 2, 1),  # 64x64
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),  # 32x32 you can change the size of the image to 64x64 by changing the kernel size and stride 3, 1
        )
        self.residual = nn.ModuleList([ResidualBlock(hidden_channels) for _ in range(res_nums)])       
        self.conv4 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)     
        self.leakrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.downsample(x)
        for block in self.residual:
            x = block(x)
        x = self.conv4(x)
        return self.leakrelu(x)
    
if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512)
    model = Encoder(3, 128, 256, 4)
    print(model(x).shape)#1,128,32,32