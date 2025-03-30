import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torch.nn.functional as F
from tqdm import tqdm
import os

# 导入模型（确保你已经将 VQVAE 和 PixelCNN 的实现做了解耦，VQVAE 不再包含 PixelCNN 部分）
from VQVAE import VQVAE, VQVAE_EMA, VQVAE_Gumbel
from PixelCNN import GatedPixelCNN

# 配置
config = {
    # 基础配置
    "batch_size": 512,
    "vqvae_epochs": 50,
    "pixelcnn_epochs": 100,
    "learning_rate": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "in_channels": 3,         # 输入图像通道数（例如 CIFAR-100 为 3）
    "out_channels": 64,       # 输出图像通道数
    "hidden_channels": 128,   # 隐藏层通道数
    "res_nums": 2,            # 残差块数量
    "n_e": 512,               # 代码簿大小（嵌入向量数量）
    "e_dim": 64,              # 嵌入向量维度
    "beta": 0.25,             # VQ 损失权重参数
    "image_size": (8, 8),     # 特征图大小
    "result_dir": "results",  # 结果保存目录
    "max_grad_norm": 10.0,    # 梯度裁剪最大范数

    # EMA 配置（用于 VQVAE_EMA）
    "ema_decay": 0.99,
    "ema_epsilon": 1e-5,

    # Gumbel-Softmax 配置（用于 VQVAE_Gumbel）
    "initial_temp": 1.0,
    "min_temp": 0.5,
    "anneal_rate": 1e-5,
    
    "n_layers": 15,  # PixelCNN 的层数
}

# 创建结果保存文件夹
os.makedirs(os.path.join(config["result_dir"], "model"), exist_ok=True)
os.makedirs(os.path.join(config["result_dir"], "image"), exist_ok=True)
os.makedirs(os.path.join(config["result_dir"], "sample"), exist_ok=True)

# 数据集加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip()
])
train_dataset = datasets.CIFAR100(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=4
)

# 构建模型：注意这里 VQVAE 不再包含 PixelCNN 部分
vqvae = VQVAE(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        hidden_channels=config["hidden_channels"],
        res_nums=config["res_nums"],
        n_e=config["n_e"],
        e_dim=config["e_dim"],
        beta=config["beta"],
        device=config["device"],
    ).to(config["device"])

vqvae_ema = VQVAE_EMA(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        hidden_channels=config["hidden_channels"],
        res_nums=config["res_nums"],
        n_e=config["n_e"],
        e_dim=config["e_dim"],
        beta=config["beta"],
        decay=0.99,
        epsilon=1e-5,
        device=config["device"],
    ).to(config["device"])

vqvae_gumbel = VQVAE_Gumbel(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        hidden_channels=config["hidden_channels"],
        res_nums=config["res_nums"],
        n_e=config["n_e"],
        e_dim=config["e_dim"],
        beta=config["beta"],
        device=config["device"],
        initial_temp=config["initial_temp"],
        min_temp=config["min_temp"],
        anneal_rate=config["anneal_rate"],
    ).to(config["device"])

# 单独构建 PixelCNN 模块
pixelcnn = GatedPixelCNN(
    n_embeddings=config["n_e"],
    dim=config["e_dim"],
    n_layers=config["n_layers"]
).to(config["device"])

# 优化器与学习率调度器（分别为 VQVAE 与 PixelCNN 定义）
optimizer_vqvae = optim.AdamW(vqvae.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
scheduler_vqvae = optim.lr_scheduler.LambdaLR(optimizer_vqvae, lr_lambda=lambda epoch: min(epoch / 10, 1.0))
optimizer_pixelcnn = optim.AdamW(pixelcnn.parameters(), lr=config["learning_rate"], weight_decay=1e-4)

######################################
# 第一阶段：单独训练 VQVAE（不涉及 PixelCNN 部分）
######################################
def train_vqvae():
    print("开始训练 VQVAE ...")
    for epoch in range(1, config["vqvae_epochs"] + 1):
        vqvae.train()
        running_loss = 0.0
        running_recon = 0.0
        running_quant = 0.0
        running_perplexity = 0.0

        loop = tqdm(train_loader, desc=f"VQVAE Epoch [{epoch}/{config['vqvae_epochs']}]")
        for batch_idx, (data, _) in enumerate(loop):
            data = data.to(config["device"])
            optimizer_vqvae.zero_grad()
            # vqvae 返回：total_loss, recon, recon_loss, quant_loss, perplexity, quantized, min_encoding_indices
            total_loss, recon, recon_loss, quant_loss, perplexity, _, _ = vqvae(data)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vqvae.parameters(), config["max_grad_norm"])
            optimizer_vqvae.step()

            running_loss += total_loss.item()
            running_recon += recon_loss.item()
            running_quant += quant_loss.item()
            running_perplexity += perplexity.item()

            loop.set_postfix(
                total=f"{total_loss.item():.4f}",
                recon=f"{recon_loss.item():.4f}",
                quant=f"{quant_loss.item():.4f}",
                ppl=f"{perplexity.item():.2f}"
            )

        scheduler_vqvae.step()

        avg_loss = running_loss / len(train_loader)
        avg_recon = running_recon / len(train_loader)
        avg_quant = running_quant / len(train_loader)
        avg_ppl = running_perplexity / len(train_loader)
        print(f"\nVQVAE Epoch {epoch} Summary:")
        print(f"Total Loss: {avg_loss:.4f} | Recon Loss: {avg_recon:.4f} | Quant Loss: {avg_quant:.4f}")
        print(f"Perplexity: {avg_ppl:.2f}\n")

        # 保存 VQVAE 模型
        torch.save(vqvae.state_dict(), os.path.join(config["result_dir"], "model", f"vqvae_epoch_{epoch}.pth"))

        # 保存重构图像样本
        with torch.no_grad():
            vqvae.eval()
            sample = data[:8]
            _, recon_sample, _, _, _, _, _ = vqvae(sample)
            comparison = torch.cat([sample, recon_sample], dim=0) * 0.5 + 0.5
            utils.save_image(comparison.cpu(), os.path.join(config["result_dir"], "image", f"recon_epoch_{epoch}.png"), nrow=8)
    print("VQVAE 训练完成！")

######################################
# 第二阶段：单独训练 PixelCNN
######################################
def train_pixelcnn():
    print("开始训练 PixelCNN ...")
    # 固定 VQVAE 参数，不再更新
    vqvae_path = r"VectorQuantization\results\model\vqvae_epoch_50.pth"
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=config["device"]))
    vqvae.eval()  # 固定 VQVAE 参数，不再更新
    for epoch in range(1, config["pixelcnn_epochs"] + 1):
        pixelcnn.train()
        running_pixelcnn_loss = 0.0

        loop = tqdm(train_loader, desc=f"PixelCNN Epoch [{epoch}/{config['pixelcnn_epochs']}]")
        for batch_idx, (data, _) in enumerate(loop):
            data = data.to(config["device"])
            with torch.no_grad():
                # 利用固定的 VQVAE 得到离散索引和量化后的特征图
                # vqvae 返回：total_loss, recon, recon_loss, quant_loss, perplexity, quantized, min_encoding_indices
                _, _, _, _, _, quantized, min_encoding_indices = vqvae(data)
            # 获取 latent 尺寸（例如 8x8）
            B, _, H_latent, W_latent = quantized.shape
            # 重塑为 [B, H_latent, W_latent]
            indices = min_encoding_indices.view(B, H_latent, W_latent)
            
            optimizer_pixelcnn.zero_grad()
            logits = pixelcnn(indices)  # logits: [B, n_e, H, W]
            loss_pixelcnn = F.cross_entropy(logits.reshape(-1, config["n_e"]), indices.reshape(-1))
            loss_pixelcnn.backward()
            optimizer_pixelcnn.step()

            running_pixelcnn_loss += loss_pixelcnn.item()
            loop.set_postfix(pixelcnn_loss=f"{loss_pixelcnn.item():.4f}")

        avg_pixelcnn_loss = running_pixelcnn_loss / len(train_loader)
        print(f"\nPixelCNN Epoch {epoch} Average Loss: {avg_pixelcnn_loss:.4f}\n")

        # 每个 epoch 保存一次 PixelCNN 模型
        torch.save(pixelcnn.state_dict(), os.path.join(config["result_dir"], "model", f"pixelcnn_epoch_{epoch}.pth"))

        # 使用 PixelCNN 与 VQVAE 的 Decoder 生成样本
        with torch.no_grad():
            pixelcnn.eval()
            # 先用 PixelCNN 生成离散索引
            prior_sample = pixelcnn.generate(shape=config["image_size"], batch_size=8)  # [8, H, W]
            # 将离散索引转换为 one-hot 编码，再映射为量化向量
            encodings = F.one_hot(prior_sample, num_classes=config["n_e"]).float().to(config["device"])
            # 映射到 VQVAE 的量化嵌入空间（假设 vqvae 中使用的是 VectorQuantizer，其 embedding 为 embedding.weight）
            z_q = torch.matmul(encodings.view(-1, config["n_e"]), vqvae.vq.embedding.weight)
            z_q = z_q.view(8, config["image_size"][0], config["image_size"][1], config["e_dim"]).permute(0, 3, 1, 2)
            # 使用 VQVAE 的 Decoder 生成图像
            generated = vqvae.decoder(z_q)
            utils.save_image(generated.cpu() * 0.5 + 0.5, os.path.join(config["result_dir"], "sample", f"sample_epoch_{epoch}.png"), nrow=8)
    print("PixelCNN 训练完成！")


if __name__ == "__main__":
    # 先单独训练 VQVAE
    # train_vqvae()
    # 训练完成后，再单独训练 PixelCNN
    train_pixelcnn()
