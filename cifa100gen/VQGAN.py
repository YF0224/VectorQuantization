import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder
from Decoder import Decoder
from VectorQuantization import VectorQuantizer, VectorQuantizerEMA, VectorQuantizerGumbelSoftmax
from Transformer import Transformer
from GAN import GAN


class VQGAN(nn.module):
    def __init__(self, in_channels, out_channels, hidden_channels, res_nums, n_e, e_dim, beta, device, n_head, dim_feedforward, num_layers, dropout):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels, hidden_channels, res_nums)
        self.vq = VectorQuantizer(n_e, e_dim, beta, device)
        self.transformer = Transformer(n_e, 64, e_dim, n_head, dim_feedforward, num_layers, dropout)
        self.decoder = Decoder(in_channels, out_channels, hidden_channels, res_nums)
        self.gan = GAN()
        self.device = device
        
    def forward(self, x):
        z = self.encoder(x)
        quant_loss, quantized, perplexity, _, min_encoding_indices = self.vq(z)
        recon = self.decoder(quantized)
        recon_loss = F.mse_loss(recon, x)
        transformer_loss = self.transformer_loss(min_encoding_indices, quantized)
        total_loss = recon_loss + quant_loss + transformer_loss
        return total_loss, recon, recon_loss, quant_loss, transformer_loss, perplexity