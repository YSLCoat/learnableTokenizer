import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio

import torch
import numpy as np
import scipy.io
from skimage.color import rgb2lab
import matplotlib.pyplot as plt



class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        # read the image and the mask (label), and store them in the form of numpy arrays (for the albumentations library)
        image = np.array(Image.open(img_path).convert("RGB")) # we might not need to do this because the images are loaded as RGB by default
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # convert("L") will convert the image to grayscale
        # binarize the mask
        mask[mask == 255.0] = 1.0

        # apply the transformations if they exist
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        return image, mask 
    

def convert_label(label):

    onehot = np.zeros((1, 50, label.shape[0], label.shape[1])).astype(np.float32)

    ct = 0
    for t in np.unique(label).tolist():
        if ct >= 50:
            break
        else:
            onehot[:, ct, :, :] = (label == t)
        ct = ct + 1

    return onehot


class BSDS:
    def __init__(self, root, split="train", color_transforms=None, geo_transforms=None):
        self.gt_dir = os.path.join(root, "BSDS500/data/ground_truth", split)
        self.img_dir = os.path.join(root, "BSDS500/data/images", split)

        self.index = os.listdir(self.gt_dir)

        self.color_transforms = color_transforms
        self.geo_transforms = geo_transforms


    def __getitem__(self, idx):
        idx = self.index[idx][:-4]
        gt = scipy.io.loadmat(os.path.join(self.gt_dir, idx+".mat"))
        t = np.random.randint(0, len(gt['groundTruth'][0]))
        gt = gt['groundTruth'][0][t][0][0][0]

        img = rgb2lab(plt.imread(os.path.join(self.img_dir, idx+".jpg")))

        gt = gt.astype(np.int64)
        img = img.astype(np.float32)

        if self.color_transforms is not None:
            img = self.color_transforms(img)

        if self.geo_transforms is not None:
            img, gt = self.geo_transforms([img, gt])

        gt = convert_label(gt)
        gt = torch.from_numpy(gt)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        return img, gt.reshape(50, -1).float()


    def __len__(self):
        return len(self.index)
    
    
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
        
        # Load image
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        # Ensure the image is resized to a fixed size
        if self.transform:
            image = self.transform(image)
        
        # Load ground truth segmentation
        gt_name = os.path.join(self.ground_truth_dir, self.image_files[idx].replace('.jpg', '.mat'))
        gt_data = sio.loadmat(gt_name)
        t = np.random.randint(0, len(gt_data['groundTruth'][0]))  # Select random segmentation labels for each image.
        ground_truth = gt_data['groundTruth'][0][t][0][0][0]
        segmentation = convert_label(ground_truth)
        
        # Reduce the one-hot encoding to a single channel using argmax
        segmentation = np.argmax(segmentation, axis=1).squeeze(0)
        
        # Resize the segmentation mask to match the image size
        segmentation = Image.fromarray(segmentation.astype(np.uint8))
        segmentation = segmentation.resize((224, 224), Image.NEAREST)
        
        # Convert segmentation to a torch tensor
        segmentation = torch.tensor(np.array(segmentation), dtype=torch.long)
        
        return image, segmentation
