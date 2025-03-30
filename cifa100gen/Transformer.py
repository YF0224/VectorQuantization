import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, e_dim):
        super().__init__()
        # 初始化位置编码矩阵
        pe = torch.zeros(seq_len, e_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, e_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / e_dim))
        
        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置：sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置：cos
        
        # 注册为缓冲区
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, seq_len, e_dim]

    def forward(self, x):
        # 将位置编码添加到输入 x 上
        return x + self.pe[:, :x.size(1)]  # 确保位置编码的长度与输入序列长度匹配

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head  # 每个头的维度

        # 线性变换层
        self.Wq = nn.Linear(d_model, d_model)  # 查询（Query）变换
        self.Wk = nn.Linear(d_model, d_model)  # 键（Key）变换
        self.Wv = nn.Linear(d_model, d_model)  # 值（Value）变换
        self.Wo = nn.Linear(d_model, d_model)  # 输出变换

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, Q, K, V, mask):
        batch_size = Q.size(0)

        # 线性变换
        Q = self.Wq(Q)  # [batch_size, seq_len, d_model]
        K = self.Wk(K)  # [batch_size, seq_len, d_model]
        V = self.Wv(V)  # [batch_size, seq_len, d_model]

        # 多头分割
        Q = Q.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)  # [batch_size, n_head, seq_len, head_dim]
        K = K.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)  # [batch_size, n_head, seq_len, head_dim]
        V = V.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)  # [batch_size, n_head, seq_len, head_dim]

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # [batch_size, n_head, seq_len, seq_len]

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # 将掩码为 0 的位置设置为负无穷

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, n_head, seq_len, seq_len]

        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, n_head, seq_len, head_dim]

        # 多头合并
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, seq_len, d_model]

        # 输出变换
        output = self.Wo(attn_output)  # [batch_size, seq_len, d_model]

        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 或 nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1):
        super().__init__()
        # 多头注意力层
        self.self_attn = MaskedMultiHeadAttention(d_model, n_head)
        
        # 前馈网络层
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力子层（带残差连接）
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 前馈网络子层（带残差连接）
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=512, n_head=8, 
                 dim_feedforward=2048, num_layers=6, dropout=0.1):
        super().__init__()

        # 位置编码
        self.positional_encoding = PositionalEncoding(seq_len, d_model)
        
        # Transformer 块堆叠
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 输入嵌入 + 位置编码
        x = self.embedding(x)  # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 通过多个 Transformer 块
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # 输出投影和 softmax
        x = self.linear(x)  # [batch_size, seq_len, vocab_size]
        x = self.softmax(x)
        
        return x
    
# 定义参数
batch_size = 2
seq_len = 10
vocab_size = 10000
d_model = 512
n_head = 8
dim_feedforward = 2048
num_layers = 6

x = torch.rand(1, 3, 32, 32)


# 创建掩码（下三角掩码，用于自回归任务）
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)  # [1, seq_len, seq_len]

# 初始化模型
model = Transformer(
    vocab_size=vocab_size,
    seq_len=seq_len,
    d_model=d_model,
    n_head=n_head,
    dim_feedforward=dim_feedforward,
    num_layers=num_layers
)

# 前向传播
output = model(x, mask)

# 打印形状
print("输入形状:", x.shape)
print("输出形状:", output.shape)