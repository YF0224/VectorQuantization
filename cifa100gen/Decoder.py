import torch
import torch.nn as nn
from ResidualBlock import ResidualBlock

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, res_nums):
        super().__init__()
        self.conv1 = nn.Conv2d(out_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.residual = nn.ModuleList([ResidualBlock(hidden_channels) for _ in range(res_nums)])
        self.convT1 = nn.ConvTranspose2d(hidden_channels, hidden_channels//2, kernel_size=4, stride=2, padding=1)  # 8x8 → 16x16
        self.convT2 = nn.ConvTranspose2d(hidden_channels//2, hidden_channels//4, kernel_size=4, stride=2, padding=1)  # 16x16 → 32x32
        self.conv2 = nn.Conv2d(hidden_channels//4, in_channels, kernel_size=3, stride=1, padding=1)
        self.leakrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leakrelu(self.conv1(x))  # [B, hidden_channels, 8, 8]
        for block in self.residual:
            x = block(x)                  # [B, hidden_channels, 8, 8]
        x = self.leakrelu(self.convT1(x)) # [B, hidden_channels//2, 16, 16]
        x = self.leakrelu(self.convT2(x)) # [B, hidden_channels//4, 32, 32]
        x = self.conv2(x)                 # [B, out_channels, 32, 32]
        return x

if __name__ == '__main__':
    # 测试输入：8x8特征图（例如VQ-VAE的编码器输出）
    x = torch.randn(4, 256, 8, 8)  # Batch=4, in_channels=256, 8x8
    decoder = Decoder(in_channels=3, out_channels=256, hidden_channels=256, res_nums=4)
    output = decoder(x)
    print(output.shape)  # 输出: torch.Size([4, 3, 32, 32])