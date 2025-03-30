import torch
from VQVAE import VQVAE
from torchvision.utils import save_image

# 定义模型参数
config = {
    "in_channels": 3,
    "out_channels": 32,
    "hidden_channels": 256,
    "res_nums": 2,
    "n_e": 512,
    "e_dim": 32,
    "beta": 0.25,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# 初始化模型
model = VQVAE(
    in_channels=config["in_channels"],
    out_channels=config["out_channels"],
    hidden_channels=config["hidden_channels"],
    res_nums=config["res_nums"],
    n_e=config["n_e"],
    e_dim=config["e_dim"],
    beta=config["beta"],
    device=config["device"]
).to(config["device"])

# 加载训练好的权重
model_path = "D:/OneDrive/桌面/VectorQuantization/VQ/result10/model/vqvae_epoch_100.pth"
model.load_state_dict(torch.load(model_path, map_location=config["device"]))
model.eval()

# 生成图片
num_samples = 8
with torch.no_grad():
    generated_images = model.generate(num_samples)

# 反归一化并保存图片
generated_images = generated_images * 0.5 + 0.5
save_image(generated_images.cpu(), "generated_samples.png", nrow=4)