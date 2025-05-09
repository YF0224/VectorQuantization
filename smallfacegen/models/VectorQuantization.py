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
        bt, ch, h, w = z.shape
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
        min_encoding_indices = min_encoding_indices.squeeze(1)
        min_encoding_indices = min_encoding_indices.view(bt, h, w)
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
        bt, ch, h, w = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        z_flattened = z.view(-1, self.e_dim)  # [B*H*W, C]

        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
                    torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # EMA更新逻辑略...

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(bt, h, w, ch)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        commitment_loss = F.mse_loss(z_q.detach(), z.permute(0, 3, 1, 2)) * self.beta
        loss = commitment_loss
        z_q = z.permute(0, 3, 1, 2) + (z_q - z.permute(0, 3, 1, 2)).detach()

        avg_probs = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        min_encoding_indices = min_encoding_indices.squeeze(1).view(bt, h, w)
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
        bt, ch, h, w = z.shape
        if current_step is not None:
            self.temp = max(self.min_temp, self.initial_temp * np.exp(-self.anneal_rate * current_step))

        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        logits = -d

        if self.training:
            noise = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
            logits = logits + gumbel_noise

        soft_encodings = F.softmax(logits / self.temp, dim=1)
        z_q = torch.matmul(soft_encodings, self.embedding.weight).view(bt, h, w, ch)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        z_perm = z.permute(0, 3, 1, 2)
        commitment_loss = self.beta * F.mse_loss(z_q.detach(), z_perm)
        codebook_loss = F.mse_loss(z_perm, z_q.detach())
        loss = codebook_loss + commitment_loss

        if not self.training:
            min_encoding_indices = torch.argmax(soft_encodings, dim=1)
            min_encodings = F.one_hot(min_encoding_indices, self.n_e).float()
        else:
            min_encoding_indices = torch.argmax(soft_encodings, dim=1)
            min_encodings = soft_encodings  # soft one-hot

        avg_probs = torch.mean(soft_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        min_encoding_indices = min_encoding_indices.view(bt, h, w)
        return loss, z_q, perplexity, min_encodings, min_encoding_indices
