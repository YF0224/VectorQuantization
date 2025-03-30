import torch
import torch.nn as nn
import torch.nn.functional as F
from ResidualBlock import ResidualBlock

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, res_nums):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels // 4, kernel_size=4, stride=2, padding=1) # 32*32->16*16
        self.conv2 = nn.Conv2d(hidden_channels // 4, hidden_channels // 2, kernel_size=4, stride=2, padding=1) # 16*16->8*8
        self.conv3 = nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=3, stride=1, padding=1) # 8*8->8*8
        self.residual = nn.ModuleList([ResidualBlock(hidden_channels) for _ in range(res_nums)])
        self.leakrelu = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1) # the same size
        
    def forward(self, x):
        x = self.leakrelu(self.conv1(x))
        x = self.leakrelu(self.conv2(x))
        x = self.leakrelu(self.conv3(x))
        for block in self.residual:
            x = block(x)
        x = self.conv4(x)
        return x


if __name__ == '__main__':
    x = torch.rand(1, 3, 32, 32)
    model = Encoder(3, 64, 256, 4)
    print(model(x).shape)
        