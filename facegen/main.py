from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import torch.nn.functional as F
from tqdm import tqdm
import os
from VQVAE import VQVAE, VQVAE_EMA, VQVAE_Gumbel

# 配置
config = {
    # 基础配置
    "batch_size": 256,
    "epochs": 100,
    "learning_rate": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "in_channels": 3,  # 输入图像的通道数（CIFAR-100 是 3 通道）
    "out_channels": 64,  # 输出图像的通道数
    "hidden_channels": 256,  # 隐藏层通道数
    "res_nums": 4,  # 残差块的数量
    "n_e": 1024,  # 代码簿大小（嵌入向量的数量）
    "e_dim": 64,  # 嵌入向量的维度
    "beta": 0.25,  # VQ 损失的权重参数
    "image_size": (32, 32),  # 特征图的大小
    "result_dir": "resultCeleba",  # 结果保存目录
    "max_grad_norm": 10.0,  # 梯度裁剪的最大范数

    # EMA 配置（用于 VQVAE_EMA）
    "ema_decay": 0.99,  # EMA 衰减率
    "ema_epsilon": 1e-5,  # EMA 的 epsilon（用于数值稳定性）

    # Gumbel-Softmax 配置（用于 VQVAE_Gumbel）
    "initial_temp": 1.0,  # 初始温度
    "min_temp": 0.5,  # 最小温度
    "anneal_rate": 1e-5,  # 温度衰减率
    
    "n_layers": 15,  # PixelCNN 的层数
}

# 创建结果保存文件夹
os.makedirs(os.path.join(config["result_dir"], "model"), exist_ok=True)
os.makedirs(os.path.join(config["result_dir"], "image"), exist_ok=True)
os.makedirs(os.path.join(config["result_dir"], "sample"), exist_ok=True)

class CelebAHQDataset(Dataset):
    def __init__(self, root, img_size=512):
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
    
root = r""#your path to CelebA HQ dataset
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
        device=config["device"],
        n_layers=config["n_layers"],
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
        n_layers=config["n_layers"]
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
        n_layers=config["n_layers"]
    ).to(config["device"])

optimizer = optim.AdamW(vqvae.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(epoch / 10, 1.0))


def train():
    for epoch in range(1, config["epochs"] + 1):
        vqvae.train()
        running_loss = 0.0
        running_recon = 0.0
        running_quant = 0.0
        running_pixelcnn = 0.0
        running_perplexity = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{config['epochs']}]")
        for batch_idx, data in enumerate(loop):
            data = data.to(config["device"])
            
            optimizer.zero_grad()
            total_loss, recon, recon_loss, quant_loss, pixelcnn_loss, perplexity = vqvae(data)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vqvae.parameters(), config["max_grad_norm"])
            optimizer.step()
            
            running_loss += total_loss.item()
            running_recon += recon_loss.item()
            running_quant += quant_loss.item()
            running_pixelcnn += pixelcnn_loss.item()
            running_perplexity += perplexity.item()
            
            loop.set_postfix(
                total=f"{total_loss.item():.4f}",
                recon=f"{recon_loss.item():.4f}",
                quant=f"{quant_loss.item():.4f}",
                pixelcnn=f"{pixelcnn_loss.item():.4f}",
                ppl=f"{perplexity.item():.2f}"
            )
        
        scheduler.step()
        
        avg_loss = running_loss / len(train_loader)
        avg_recon = running_recon / len(train_loader)
        avg_quant = running_quant / len(train_loader)
        avg_pixelcnn = running_pixelcnn / len(train_loader)
        avg_ppl = running_perplexity / len(train_loader)
        print(f"\nEpoch {epoch} Summary:")
        print(f"Total Loss: {avg_loss:.4f} | Recon Loss: {avg_recon:.4f} | Quant Loss: {avg_quant:.4f} | PixelCNN Loss: {avg_pixelcnn:.4f}")
        print(f"Perplexity: {avg_ppl:.2f}\n")
        
        # 保存模型
        torch.save(vqvae.state_dict(), os.path.join(config["result_dir"], "model", f"vqvae_epoch_{epoch}.pth"))
        
        # 生成样本
        with torch.no_grad():
            vqvae.eval()
            sample = data[:8]
            _, recon_sample, _, _, _, _ = vqvae(sample)
            comparison = torch.cat([sample, recon_sample], dim=0) * 0.5 + 0.5
            utils.save_image(comparison.cpu(), os.path.join(config["result_dir"], "image", f"recon_epoch_{epoch}.png"), nrow=8)
            
            generated = vqvae.generate(num_samples=8, shape=config["image_size"])
            utils.save_image(generated.cpu() * 0.5 + 0.5, os.path.join(config["result_dir"], "sample", f"sample_epoch_{epoch}.png"), nrow=8)
    
    print("训练完成！")

if __name__ == "__main__":
    train()