import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import yaml
from models.VQVAE import VQVAE
from models.PixelCNN import GatedPixelCNN
from utils.data_loader import get_celeba_dataset
from utils.logger import Logger

def main():
    # 加载配置
    with open("configs/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)


    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志记录器
    logger = Logger(config)

    # 数据加载器
    transform = transforms.Compose([
        transforms.Resize(config["data"]["image_size"]),
        transforms.CenterCrop(config["data"]["image_size"]),
        transforms.ToTensor(),
    ])

    dataset = get_celeba_dataset(config["data"]["dataset_path"], transform)
    dataloader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=4)

    # 初始化 VQVAE
    vqvae = VQVAE(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        hidden_channels=config["model"]["hidden_channels"],
        res_nums=config["model"]["residual_blocks"],
        n_e=config["model"]["n_e"],
        e_dim=config["model"]["e_dim"],
        beta=config["model"]["beta"],
        device=device
    ).to(device)

    vqvae_optimizer = optim.Adam(vqvae.parameters(), lr=float(config["train"]["lr"]))

    start_epoch = 0
    num_epochs = config["train"]["epochs"]
    save_every = config["train"]["save_every"]

    # ============ 第一阶段：训练 VQVAE ============
    for epoch in range(start_epoch, num_epochs):
        vqvae.train()
        total_loss = 0
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)

            vqvae_optimizer.zero_grad()
            loss, recon, recon_loss, vq_loss, perplexity, _, _ = vqvae(batch)
            loss.backward()
            vqvae_optimizer.step()

            total_loss += loss.item()

            if i % config["log"]["log_interval"] == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, VQ: {vq_loss.item():.4f}, Perp: {perplexity.item():.4f}")

                logger.log_scalar("loss/total", loss.item(), epoch * len(dataloader) + i)
                logger.log_scalar("loss/recon", recon_loss.item(), epoch * len(dataloader) + i)
                logger.log_scalar("loss/vq", vq_loss.item(), epoch * len(dataloader) + i)
                logger.log_scalar("perplexity", perplexity.item(), epoch * len(dataloader) + i)

        # 保存样本和模型
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            vqvae.eval()
            with torch.no_grad():
                sample = batch[:16]
                recon = vqvae.generate(sample)
                vutils.save_image(torch.cat([sample, recon], dim=0),
                                  os.path.join("outputs/samples", f"recon_epoch_{epoch+1}.png"),
                                  nrow=8, normalize=True)

            torch.save(vqvae.state_dict(), os.path.join("outputs/checkpoints", f"vqvae_epoch_{epoch+1}.pth"))
            print(f"Saved VQVAE model and samples at epoch {epoch+1}")

    # 设置 VQVAE 为评估模式
    vqvae.eval()

    # ============ 第二阶段：训练 PixelCNN ============
    print("\nStarting PixelCNN training...")

    pixelcnn = GatedPixelCNN(
        n_embeddings=config["model"]["n_e"],
        dim=config["model"]["e_dim"],
        n_layers=config["pixelcnn"]["n_layers"]
    ).to(device)

    pixelcnn_optimizer = optim.Adam(pixelcnn.parameters(), lr=config["pixelcnn"]["lr"])
    pixelcnn_epochs = config["pixelcnn"]["epochs"]

    for epoch in range(pixelcnn_epochs):
        total_pixelcnn_loss = 0
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            with torch.no_grad():
                _, _, _, _, _, _, indices = vqvae(batch)
                indices = indices.view(indices.size(0), *config["pixelcnn"]["input_shape"])  # e.g. (B, 32, 32)

            logits = pixelcnn(indices)
            loss = nn.CrossEntropyLoss()(logits, indices)

            pixelcnn_optimizer.zero_grad()
            loss.backward()
            pixelcnn_optimizer.step()

            total_pixelcnn_loss += loss.item()

            if i % config["log"]["log_interval"] == 0:
                print(f"[PixelCNN] Epoch [{epoch+1}/{pixelcnn_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
                logger.log_scalar("pixelcnn/loss", loss.item(), epoch * len(dataloader) + i)

        print(f"Epoch [{epoch+1}/{pixelcnn_epochs}] PixelCNN Avg Loss: {total_pixelcnn_loss / len(dataloader):.4f}")
        torch.save(pixelcnn.state_dict(), os.path.join("outputs/checkpoints", f"pixelcnn_epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # 可选
    main()