import torch
import torch.nn as nn  
import torch.nn.functional as F
from ResidualBlock import ResidualBlock
# because the decoder is the reverse of the encoder, so i make the in_channels and out_channels exchange, just to simple the vqvae 
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, res_nums):
        super().__init__()
        self.init_conv = nn.Conv2d(out_channels, hidden_channels, 3, 1, 1)
        
        # 残差模块
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(res_nums)]
        )
        
        # 上采样模块 (32x32 -> 512x512)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels//2, 4, 2, 1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels//2, hidden_channels//4, 4, 2, 1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels//4, hidden_channels//8, 4, 2, 1),  # 256x256
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels//8, hidden_channels//16, 4, 2, 1),  # 512x512
        )
        
        self.final_conv = nn.Conv2d(hidden_channels//16, in_channels, 3, 1, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_blocks(x)
        x = self.upsample(x)
        x = self.final_conv(x)
        return self.tanh(x)
    
if __name__ == '__main__':

    x = torch.randn(4, 256, 32, 32)  
    decoder = Decoder(in_channels=3, out_channels=256, hidden_channels=256, res_nums=4)
    output = decoder(x)
    print(output.shape)  