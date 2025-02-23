import os
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import numpy as np
import scipy.io as sio


# -------------------------------
# BSDS500 Dataset
# -------------------------------
class BSDS500Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images', split)
        self.ground_truth_dir = os.path.join(root_dir, 'ground_truth', split)
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image
    