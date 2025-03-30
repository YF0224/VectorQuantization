import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from tqdm import tqdm
import os
from PIL import Image

from VQVAE import VQVAE, VQVAE_EMA, VQVAE_Gumbel

# 配置
config = {
    # 基础配置
    "batch_size": 128,
    "epochs": 100,
    "learning_rate": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "in_channels": 3,  # 输入图像的通道数（CIFAR-100 是 3 通道）
    "out_channels": 64,  # 输出图像的通道数
    "hidden_channels": 128,  # 隐藏层通道数
    "res_nums": 2,  # 残差块的数量
    "n_e": 512,  # 代码簿大小（嵌入向量的数量）
    "e_dim": 64,  # 嵌入向量的维度
    "beta": 0.25,  # VQ 损失的权重参数
    "image_size": (8, 8),  # 特征图的大小
    "result_dir": "resultCele",  # 结果保存目录
    "max_grad_norm": 10.0,  # 梯度裁剪的最大范数

    # EMA 配置（用于 VQVAE_EMA）
    "ema_decay": 0.99,  # EMA 衰减率
    "ema_epsilon": 1e-5,  # EMA 的 epsilon（用于数值稳定性）

    # Gumbel-Softmax 配置（用于 VQVAE_Gumbel）
    "initial_temp": 1.0,  # 初始温度
    "min_temp": 0.5,  # 最小温度
    "anneal_rate": 1e-5,  # 温度衰减率
}

# 创建结果保存文件夹
os.makedirs(os.path.join(config["result_dir"], "model"), exist_ok=True)
os.makedirs(os.path.join(config["result_dir"], "image"), exist_ok=True)
os.makedirs(os.path.join(config["result_dir"], "sample"), exist_ok=True)

# 数据集加载
class CelebAHQDataset(Dataset):
    def __init__(self, root, img_size=32):
        self.img_paths = [os.path.join(root, f) for f in os.listdir(root) 
                         if f.endswith(('.jpg', '.png'))]
        self.transform = transforms.Compose([
            transforms.Resize(img_size + 64),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img)
    
root = r""#your CelebA HQ dataset path
dataset = CelebAHQDataset(root)
train_loader = DataLoader(dataset, batch_size=config["batch_size"], 
                            shuffle=True, num_workers=8, pin_memory=True)


vqvae = VQVAE(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        hidden_channels=config["hidden_channels"],
        res_nums=config["res_nums"],
        n_e=config["n_e"],
        e_dim=config["e_dim"],
        beta=config["beta"],
        device=config["device"]
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
        device=config["device"]
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
        anneal_rate=config["anneal_rate"]
    ).to(config["device"])

def train():
    model = vqvae_ema

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: min(epoch / 10, 1.0)  # 10 epoch warmup
    )

    # 训练循环
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_recon = 0.0
        total_quant = 0.0
        total_loss = 0.0
        total_perplexity = 0.0
        
        # 使用tqdm进度条
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{config['epochs']}]")
        for batch_idx, data in enumerate(loop):
            data = data.to(config["device"])
            
            # 前向传播
            quant_loss, recon, perplexity = model(data)  # 注意输出顺序
            
            # 检查 recon 和 data 的形状
            if recon.shape != data.shape:
                raise RuntimeError(f"Shape mismatch: recon {recon.shape}, data {data.shape}")
            
            # 计算重建损失
            recon_loss = F.mse_loss(recon, data, reduction='mean')  # 确保 reduction='mean'
            batch_total_loss = recon_loss + quant_loss
            
            # 反向传播
            optimizer.zero_grad()
            batch_total_loss.backward()  # 确保 batch_total_loss 是标量
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.get("max_grad_norm", 10.0)
            )
            optimizer.step()
            
            # 累计统计
            total_recon += recon_loss.item()
            total_quant += quant_loss.item()
            total_loss += batch_total_loss.item()
            total_perplexity += perplexity.item()
            
            # 更新进度条
            loop.set_postfix(
                recon=f"{recon_loss.item():.4f}",
                quant=f"{quant_loss.item():.4f}",
                total=f"{batch_total_loss.item():.4f}",
                ppl=f"{perplexity.item():.2f}"
            )
        
        # 更新学习率
        scheduler.step()
        
        # 计算epoch平均值
        avg_recon = total_recon / len(train_loader)
        avg_quant = total_quant / len(train_loader)
        avg_total = total_loss / len(train_loader)
        avg_ppl = total_perplexity / len(train_loader)
        
        # 打印epoch统计
        print(f"\nEpoch {epoch} Summary:")
        print(f"Total Loss: {avg_total:.4f} | Recon: {avg_recon:.4f} | Quant: {avg_quant:.4f}")
        print(f"Perplexity: {avg_ppl:.2f}\n")
        
        # 保存模型（每个epoch）
        torch.save(
            model.state_dict(),
            os.path.join(config["result_dir"], "model", f"vqvae_ema_epoch_{epoch}.pth")
        )
        
        # 生成可视化样本
        with torch.no_grad():
            model.eval()
            # 重建对比图
            sample = data[:8]
            quant_loss, recon, _ = model(sample)  # 注意输出顺序
            comparison = torch.cat([sample, recon], dim=0) * 0.5 + 0.5
            utils.save_image(
                comparison.cpu(),
                os.path.join(config["result_dir"], "image", f"recon_epoch_{epoch}.png"),
                nrow=8
            )
            
            # 随机生成图
            generated = model.generate(num_samples=8)
            utils.save_image(
                generated.cpu() * 0.5 + 0.5,
                os.path.join(config["result_dir"], "sample", f"sample_epoch_{epoch}.png"),
                nrow=8
            )
            model.train()

    print("训练完成！")

if __name__ == "__main__":
    train()
