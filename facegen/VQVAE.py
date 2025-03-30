import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder
from Decoder import Decoder
from VectorQuantization import VectorQuantizer, VectorQuantizerEMA, VectorQuantizerGumbelSoftmax
from PixelCNN import GatedPixelCNN

class VQVAE(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, res_nums, n_e, e_dim, beta, device, n_layers):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels, hidden_channels, res_nums)
        # VectorQuantizer 返回 (loss, quantized, perplexity, min_encodings, min_encoding_indices)
        self.vq = VectorQuantizer(n_e, e_dim, beta, device)
        self.pixelcnn = GatedPixelCNN(n_e, e_dim, n_layers) 
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
        
        # Step 5: 使用 VectorQuantizer 返回的索引计算 PixelCNN 损失
        pixelcnn_loss = self.pixelcnn_loss(min_encoding_indices, quantized)  # 此处使用第二个参数仅用于获取形状
        
        # Step 6: 总损失 = 重构损失 + 量化损失 + PixelCNN 损失
        total_loss = recon_loss + quant_loss + pixelcnn_loss
        
        return total_loss, recon, recon_loss, quant_loss, pixelcnn_loss, perplexity

    def pixelcnn_loss(self, min_encoding_indices, x):
        """
        利用 VectorQuantizer 返回的离散索引计算 PixelCNN 损失，
        PixelCNN 的目标是预测每个位置上正确的离散索引。
        """
        # x 的形状为 [B, C, H, W]，而 min_encoding_indices 的形状原本为 [B*H*W, 1]
        B, _, H, W = x.shape
        # 重塑索引为 [B, H, W]
        indices = min_encoding_indices.view(B, H, W)
        
        # 将离散索引输入 PixelCNN，PixelCNN 的 forward 要求输入形状为 [B, H, W]
        logits = self.pixelcnn(indices)  # 输出 logits 形状应为 [B, n_e, H, W]
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits.reshape(-1, self.vq.n_e), indices.reshape(-1))
        return loss

    def generate(self, num_samples, shape=(32, 32)):
        with torch.no_grad():
            # Step 1: 使用 PixelCNN 生成先验采样（离散索引）
            prior_sample = self.pixelcnn.generate(shape=shape, batch_size=num_samples)  # [num_samples, H, W]

            # Step 2: 将先验采样转换为 one-hot 编码，再映射为潜在向量
            encodings = F.one_hot(prior_sample, self.vq.n_e).float().to(self.device)  # [num_samples, H, W, n_e]
            z_q = torch.matmul(encodings.view(-1, self.vq.n_e), self.vq.embedding.weight)  # [num_samples*H*W, e_dim]
            z_q = z_q.view(num_samples, shape[0], shape[1], self.vq.e_dim).permute(0, 3, 1, 2)  # [num_samples, e_dim, H, W]
            
            # Step 3: Decoder 根据潜在向量生成图像
            generated = self.decoder(z_q)
            return generated

class VQVAE_EMA(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, res_nums, n_e, e_dim, beta, decay, epsilon, device, n_layers):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels, hidden_channels, res_nums)
        self.vq = VectorQuantizerEMA(n_e, e_dim, beta, decay, epsilon, device)
        self.pixelcnn = GatedPixelCNN(n_e, e_dim, n_layers) 
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
        
        # Step 5: 使用 VectorQuantizer 返回的索引计算 PixelCNN 损失
        pixelcnn_loss = self.pixelcnn_loss(min_encoding_indices, quantized)  # 此处使用第二个参数仅用于获取形状
        
        # Step 6: 总损失 = 重构损失 + 量化损失 + PixelCNN 损失
        total_loss = recon_loss + quant_loss + pixelcnn_loss
        
        return total_loss, recon, recon_loss, quant_loss, pixelcnn_loss, perplexity

    def pixelcnn_loss(self, min_encoding_indices, x):
        """
        利用 VectorQuantizer 返回的离散索引计算 PixelCNN 损失，
        PixelCNN 的目标是预测每个位置上正确的离散索引。
        """
        # x 的形状为 [B, C, H, W]，而 min_encoding_indices 的形状原本为 [B*H*W, 1]
        B, _, H, W = x.shape
        # 重塑索引为 [B, H, W]
        indices = min_encoding_indices.view(B, H, W)
        
        # 将离散索引输入 PixelCNN，PixelCNN 的 forward 要求输入形状为 [B, H, W]
        logits = self.pixelcnn(indices)  # 输出 logits 形状应为 [B, n_e, H, W]
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits.reshape(-1, self.vq.n_e), indices.reshape(-1))
        return loss

    def generate(self, num_samples, shape=(32, 32)):
        with torch.no_grad():
            # Step 1: 使用 PixelCNN 生成先验采样（离散索引）
            prior_sample = self.pixelcnn.generate(shape=shape, batch_size=num_samples)  # [num_samples, H, W]

            # Step 2: 将先验采样转换为 one-hot 编码，再映射为潜在向量
            encodings = F.one_hot(prior_sample, self.vq.n_e).float().to(self.device)  # [num_samples, H, W, n_e]
            z_q = torch.matmul(encodings.view(-1, self.vq.n_e), self.vq.embedding.weight)  # [num_samples*H*W, e_dim]
            z_q = z_q.view(num_samples, shape[0], shape[1], self.vq.e_dim).permute(0, 3, 1, 2)  # [num_samples, e_dim, H, W]
            
            # Step 3: Decoder 根据潜在向量生成图像
            generated = self.decoder(z_q)
            return generated
        
class VQVAE_Gumbel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, res_nums, n_e, e_dim, beta, device, initial_temp, min_temp, anneal_rate, n_layers):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels, hidden_channels, res_nums)
        self.vq = VectorQuantizerGumbelSoftmax(n_e, e_dim, beta, device, initial_temp, min_temp, anneal_rate)
        self.pixelcnn = GatedPixelCNN(n_e, e_dim, n_layers) 
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
        
        # Step 5: 使用 VectorQuantizer 返回的索引计算 PixelCNN 损失
        pixelcnn_loss = self.pixelcnn_loss(min_encoding_indices, quantized)  # 此处使用第二个参数仅用于获取形状
        
        # Step 6: 总损失 = 重构损失 + 量化损失 + PixelCNN 损失
        total_loss = recon_loss + quant_loss + pixelcnn_loss
        
        return total_loss, recon, recon_loss, quant_loss, pixelcnn_loss, perplexity

    def pixelcnn_loss(self, min_encoding_indices, x):
        """
        利用 VectorQuantizer 返回的离散索引计算 PixelCNN 损失，
        PixelCNN 的目标是预测每个位置上正确的离散索引。
        """
        # x 的形状为 [B, C, H, W]，而 min_encoding_indices 的形状原本为 [B*H*W, 1]
        B, _, H, W = x.shape
        # 重塑索引为 [B, H, W]
        indices = min_encoding_indices.view(B, H, W)
        
        # 将离散索引输入 PixelCNN，PixelCNN 的 forward 要求输入形状为 [B, H, W]
        logits = self.pixelcnn(indices)  # 输出 logits 形状应为 [B, n_e, H, W]
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits.reshape(-1, self.vq.n_e), indices.reshape(-1))
        return loss

    def generate(self, num_samples, shape=(32, 32)):
        with torch.no_grad():
            # Step 1: 使用 PixelCNN 生成先验采样（离散索引）
            prior_sample = self.pixelcnn.generate(shape=shape, batch_size=num_samples)  # [num_samples, H, W]

            # Step 2: 将先验采样转换为 one-hot 编码，再映射为潜在向量
            encodings = F.one_hot(prior_sample, self.vq.n_e).float().to(self.device)  # [num_samples, H, W, n_e]
            z_q = torch.matmul(encodings.view(-1, self.vq.n_e), self.vq.embedding.weight)  # [num_samples*H*W, e_dim]
            z_q = z_q.view(num_samples, shape[0], shape[1], self.vq.e_dim).permute(0, 3, 1, 2)  # [num_samples, e_dim, H, W]
            
            # Step 3: Decoder 根据潜在向量生成图像
            generated = self.decoder(z_q)
            return generated