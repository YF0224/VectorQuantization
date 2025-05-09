import torch
import torch.nn as nn
import torch.nn.functional as F
from .Encoder import Encoder
from .Decoder import Decoder
from .VectorQuantization import VectorQuantizer, VectorQuantizerEMA, VectorQuantizerGumbelSoftmax


class VQVAE(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, res_nums, n_e, e_dim, beta, device):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels, hidden_channels, res_nums)
        # VectorQuantizer 返回 (loss, quantized, perplexity, min_encodings, min_encoding_indices)
        self.vq = VectorQuantizer(n_e, e_dim, beta, device)
        self.decoder = Decoder(in_channels, out_channels, hidden_channels, res_nums)
        self.device = device

    def forward(self, x):
        # Step 1: Encoder 得到潜在表示
        z = self.encoder(x)  # shape: [B, e_dim, H, W]

        # Step 2: Vector Quantization 得到量化结果及索引
        quant_loss, quantized, perplexity, _, min_encoding_indices = self.vq(z)

        # Step 3: Decoder 重构图像
        recon = self.decoder(quantized)

        # Step 4: 计算重构损失（例如 MSE）
        recon_loss = F.mse_loss(recon, x)

        # 总损失只包含重构损失和量化损失
        total_loss = recon_loss + quant_loss

        # 返回结果同时输出量化后的特征图和离散索引，供 PixelCNN 使用
        return total_loss, recon, recon_loss, quant_loss, perplexity, quantized, min_encoding_indices

    def generate(self, x):
        """给定输入 x，生成重构结果（用于测试 VQVAE 重构能力）"""
        with torch.no_grad():
            _, recon, _, _, _, _, _ = self.forward(x)
        return recon


class VQVAE_EMA(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, res_nums, n_e, e_dim, beta, decay, epsilon, device):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels, hidden_channels, res_nums)
        self.vq = VectorQuantizerEMA(n_e, e_dim, beta, decay, epsilon, device)
        self.decoder = Decoder(in_channels, out_channels, hidden_channels, res_nums)
        self.device = device

    def forward(self, x):
        # Step 1: Encoder 得到潜在表示
        z = self.encoder(x)  # shape: [B, e_dim, H, W]

        # Step 2: Vector Quantization 得到量化结果及索引
        quant_loss, quantized, perplexity, _, min_encoding_indices = self.vq(z)

        # Step 3: Decoder 重构图像
        recon = self.decoder(quantized)

        # Step 4: 计算重构损失（例如 MSE）
        recon_loss = F.mse_loss(recon, x)

        # 总损失只包含重构损失和量化损失
        total_loss = recon_loss + quant_loss

        # 返回结果同时输出量化后的特征图和离散索引，供 PixelCNN 使用
        return total_loss, recon, recon_loss, quant_loss, perplexity, quantized, min_encoding_indices

    def generate(self, x):
        """给定输入 x，生成重构结果（用于测试 VQVAE 重构能力）"""
        with torch.no_grad():
            _, recon, _, _, _, _, _ = self.forward(x)
        return recon


class VQVAE_Gumbel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, res_nums, n_e, e_dim, beta, device, initial_temp,
                 min_temp, anneal_rate):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels, hidden_channels, res_nums)
        self.vq = VectorQuantizerGumbelSoftmax(n_e, e_dim, beta, device, initial_temp, min_temp, anneal_rate)
        self.decoder = Decoder(in_channels, out_channels, hidden_channels, res_nums)
        self.device = device

    def forward(self, x):
        # Step 1: Encoder 得到潜在表示
        z = self.encoder(x)  # shape: [B, e_dim, H, W]

        # Step 2: Vector Quantization 得到量化结果及索引
        quant_loss, quantized, perplexity, _, min_encoding_indices = self.vq(z)

        # Step 3: Decoder 重构图像
        recon = self.decoder(quantized)

        # Step 4: 计算重构损失（例如 MSE）
        recon_loss = F.mse_loss(recon, x)

        # 总损失只包含重构损失和量化损失
        total_loss = recon_loss + quant_loss

        # 返回结果同时输出量化后的特征图和离散索引，供 PixelCNN 使用
        return total_loss, recon, recon_loss, quant_loss, perplexity, quantized, min_encoding_indices

    def generate(self, x):
        """给定输入 x，生成重构结果（用于测试 VQVAE 重构能力）"""
        with torch.no_grad():
            _, recon, _, _, _, _, _ = self.forward(x)
        return recon

