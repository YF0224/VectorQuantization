import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)

class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, n_e, e_dim, kernel):
        super().__init__()
        assert kernel % 2 == 1, "Kernel size must be odd"
        self.mask_type = mask_type

        self.class_cond_embedding = nn.Embedding(n_e, 2 * e_dim)
        
        # Vertical convolution
        kernel_vert = (kernel // 2 + 1, kernel)
        padding_vert = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(e_dim, e_dim*2, kernel_vert, 1, padding_vert)
        
        # Horizontal convolution
        kernel_horiz = (1, kernel // 2 + 1)
        padding_horiz = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(e_dim, e_dim*2, kernel_horiz, 1, padding_horiz)
        
        self.vert_to_horiz = nn.Conv2d(2*e_dim, 2*e_dim, 1)
        self.horiz_resid = nn.Conv2d(e_dim, e_dim, 1)
        self.gate = GatedActivation()

    def make_causal(self):
        with torch.no_grad():
            self.vert_stack.weight[:, :, -1, :].zero_()
            self.horiz_stack.weight[:, :, :, -1].zero_()


    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()
        
        h = self.class_cond_embedding(h)
        
        # Vertical branch
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.shape[-2], :x_v.shape[-1]]
        out_v = self.gate(h_vert + h[:, :, None, None])
        
        # Horizontal branch
        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.shape[-2]]
        v2h = self.vert_to_horiz(h_vert)
        
        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        out_h = self.horiz_resid(out) + x_h
        
        return out_v, out_h

class GatedPixelCNN(nn.Module):
    def __init__(self, n_embeddings, dim, n_layers):
        super().__init__()
        self.dim = dim
        
        self.embedding = nn.Embedding(n_embeddings, dim)
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            # 第一层使用 mask_type 'A'，之后使用 'B'
            layer_mask_type = 'A' if i == 0 else 'B'
            # 第一层使用较大卷积核，其余层使用较小卷积核
            layer_kernel = 7 if i == 0 else 3
            self.layers.append(
                GatedMaskedConv2d(layer_mask_type, n_embeddings, dim, layer_kernel)
            )
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, n_embeddings, 1)
        )

    def forward(self, x):
        # x 的形状为 [B, H, W]，其中每个元素为离散索引
        x = self.embedding(x)           # [B, H, W, dim]
        x = x.permute(0, 3, 1, 2)         # [B, dim, H, W]
        x_v, x_h = x, x
        B = x.shape[0]
        # 生成条件向量（例如全零）
        cond = torch.zeros(B, dtype=torch.long, device=x.device)
        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h, cond)  # 必须传入 cond
        return self.output_conv(x_h)

    def generate(self, shape=(8,8), batch_size=8):
        device = next(self.parameters()).device
        # 初始化全零张量
        x = torch.zeros((batch_size, *shape), dtype=torch.long, device=device)
        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x)
                probs = F.softmax(logits[:,:,i,j], dim=-1)
                x[:, i, j] = torch.multinomial(probs, 1).squeeze()
        return x
