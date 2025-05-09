import os
from PIL import Image
from torch.utils.data import Dataset


class CelebA(Dataset):
    def __init__(self, root, transform=None):
        """
        自定义 CelebA 数据集类
        :param root: 存放 CelebA 图像的目录，如 ./img_align_celeba
        :param transform: 图像转换（如 ToTensor、Resize 等）
        """
        self.root = root
        self.transform = transform

        # 收集所有图像路径（只保留 .jpg/.png）
        self.image_paths = sorted([
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if fname.lower().endswith((".jpg", ".png"))
        ])

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found in {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def get_celeba_dataset(root, transform):
    return CelebA(root=root, transform=transform)
