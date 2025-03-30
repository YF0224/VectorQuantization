import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, device):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.device = device

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
    
class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_e, e_dim, beta, decay, epsilon, device):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon
        self.device = device

        # 初始化码本
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # EMA统计量
        self.register_buffer("ema_cluster_size", torch.ones(n_e))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, z):
        # 输入形状: [B, C, H, W]
        z = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        z_flattened = z.view(-1, self.e_dim)     # [B*H*W, C]

        # 计算距离
        distances = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                    torch.matmul(z_flattened, self.embedding.weight.t())  # [B*H*W, n_e]

        # 找到最近邻索引
        min_encoding_indices = torch.argmin(distances, dim=1)
        min_encodings = F.one_hot(min_encoding_indices, self.n_e).float().to(z.device)

        # 统计每个码本向量的使用次数及输入向量的均值
        cluster_size = min_encodings.sum(0)  # [n_e]
        sum_encodings = torch.matmul(min_encodings.t(), z_flattened)  # [n_e, e_dim]

        if self.training:
            with torch.no_grad():
                # EMA 更新：原地更新统计量
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=(1 - self.decay))
                self.ema_w.mul_(self.decay).add_(sum_encodings, alpha=(1 - self.decay))

                # Laplace平滑
                n = self.ema_cluster_size.sum()
                cluster_size_normalized = (self.ema_cluster_size + self.epsilon) / (n + self.n_e * self.epsilon) * n

                # 更新码本权重
                self.embedding.weight.data.copy_(self.ema_w / cluster_size_normalized.unsqueeze(1))

        # 得到量化后的向量
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        # 修复梯度传播方向：对 z_q 使用 detach()
        commitment_loss = F.mse_loss(z_q.detach(), z.permute(0, 3, 1, 2)) * self.beta
        loss = commitment_loss

        # 直通估计器：允许梯度从 decoder 直接流向 encoder
        z_q = z.permute(0, 3, 1, 2) + (z_q - z.permute(0, 3, 1, 2)).detach()

        # 计算困惑度
        avg_probs = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

        
class VectorQuantizerGumbelSoftmax(nn.Module):
    def __init__(self, n_e, e_dim, beta, device, 
                 initial_temp=1.0, min_temp=0.5, anneal_rate=0.00003):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.device = device

        # 保存初始温度，并设置当前温度
        self.initial_temp = initial_temp
        self.temp = initial_temp
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate

        # 码本初始化
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, current_step=None):
        # 温度衰减（可选）
        if current_step is not None:
            # 根据初始温度和步数计算新的温度
            self.temp = max(self.min_temp, self.initial_temp * np.exp(-self.anneal_rate * current_step))

        # 输入形状: [B, C, H, W]
        z = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        z_flattened = z.view(-1, self.e_dim)     # [B*H*W, C]

        # 计算欧氏距离（并转为负的 logits，距离越小对应的 logit 越大）
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())  # [B*H*W, n_e]
        logits = -d

        # 仅在训练时添加 Gumbel 噪声
        if self.training:
            noise = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
            logits = logits + gumbel_noise  # 这里保证 logits 与噪声在相同设备上

        # Gumbel-Softmax 量化，softmax 中使用当前温度
        soft_encodings = F.softmax(logits / self.temp, dim=1)  # [B*H*W, n_e]
        z_q = torch.matmul(soft_encodings, self.embedding.weight).view(z.shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        # 损失计算：代码书损失和 commitment loss（均采用 detach 截断梯度）
        z_perm = z.permute(0, 3, 1, 2)
        commitment_loss = self.beta * torch.mean((z_q.detach() - z_perm) ** 2)
        codebook_loss = torch.mean((z_perm - z_q.detach()) ** 2)
        loss = codebook_loss + commitment_loss

        # 推理时采用硬量化：取 soft_encodings 中最大值对应的位置
        if not self.training:
            min_encoding_indices = torch.argmax(soft_encodings, dim=1)
            min_encodings = F.one_hot(min_encoding_indices, self.n_e).float()
        else:
            min_encodings = soft_encodings
            min_encoding_indices = None

        # 计算困惑度
        avg_probs = torch.mean(soft_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
