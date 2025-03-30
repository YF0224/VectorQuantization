import torch
import torch.nn as nn
import functools

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator
    
    参数：
        input_nc (int)    : 输入图像通道数（默认3）
        ndf (int)         : 初始卷积通道数（默认64）
        n_layers (int)    : 卷积层数（默认3）
        norm_layer (str)  : 归一化类型 ['batch', 'instance', 'none', 'act']（默认'batch'）
        use_sigmoid (bool): 最后是否添加Sigmoid（默认False）
        kernel_size (int) : 卷积核大小（默认4）
        padding (int)     : 填充大小（默认1）
    """
    def __init__(self, 
                 input_nc=3, 
                 ndf=64, 
                 n_layers=3, 
                 norm_layer='batch', 
                 use_sigmoid=False,
                 kernel_size=4,
                 padding=1):
        super().__init__()

        # 1. 配置归一化层
        if norm_layer.lower() == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm_layer.lower() == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            norm_layer = nn.Identity
            use_bias = True

        # 2. 构建网络层
        sequence = [
            nn.Conv2d(input_nc, ndf, 
                     kernel_size=kernel_size, 
                     stride=2, 
                     padding=padding),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kernel_size, 
                          stride=2,
                          padding=padding, 
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding,
                      bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # 输出层
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, 
                      kernel_size=kernel_size, 
                      stride=1, 
                      padding=padding)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

        # 3. 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化卷积和归一化层参数"""
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """输入: (B, C, H, W) → 输出: (B, 1, H', W')"""
        return self.model(x)

if __name__ == "__main__":
    # 测试用例
    disc = Discriminator(
        input_nc=3,
        ndf=64,
        n_layers=3,
        norm_layer='batch',
        use_sigmoid=False
    )
    
    x = torch.randn(2, 3, 32, 32)
    output = disc(x)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}") 
