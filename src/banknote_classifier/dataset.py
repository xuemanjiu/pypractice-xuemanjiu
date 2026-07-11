import os
import subprocess
import shutil
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ==========================================
# DATASET UTILITIES
# ==========================================
def download_dataset(data_path):
    dataset_dir = os.path.join(data_path, "bangla")
    if not os.path.exists(data_path) or not os.path.exists(dataset_dir):
        print("Dataset not found. Downloading Bangla-Money-Dataset from GitHub...")
        os.makedirs(data_path, exist_ok=True)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/nsojib/Bangla-Money-Dataset.git",
            dataset_dir
        ], check=True)
        # Remove git tracking to avoid nested git repository issues
        git_dir = os.path.join(dataset_dir, ".git")
        if os.path.exists(git_dir):
            shutil.rmtree(git_dir)
        print("Dataset downloaded successfully!")
    else:
        print("Dataset already exists.")

def get_img_info(data_dir, label_mapping):
    imgpath = []
    imglabel = []
    for root, dirs, _ in os.walk(data_dir):
        for sub_dir in dirs:
            if sub_dir in label_mapping:
                sub_dir_path = os.path.join(root, sub_dir)
                img_names = os.listdir(sub_dir_path)
                img_names = [f for f in img_names if f.endswith('.jpg')]
                for img_name in img_names:
                    imgpath.append(os.path.join(sub_dir_path, img_name))
                    imglabel.append(label_mapping[sub_dir])
    return imgpath, imglabel

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_paths)

class CustomTestDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_paths)

def denormalize_image(tensor):
    # Denormalize image using standard ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img