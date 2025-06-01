import os
import os.path as osp
import numpy as np
from skimage.io import imread
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image


class LoveDADataSet(data.Dataset):
    """
    LoveDA Dataset Loader for Domain Adaptation
    Supports both Urban and Rural domains for semantic segmentation
    
    LoveDA has 7 classes:
    0: Background
    1: Building  
    2: Road
    3: Water
    4: Barren
    5: Forest
    6: Agricultural
    """
    
    def __init__(self, root, domain='Urban', split='Train', max_iters=None, 
                 augmentations=None, img_size=(512, 512), 
                 mean=(123.675, 116.28, 103.53), scale=True, mirror=True, 
                 ignore_label=255):
        """
        Args:
            root: path to LoveDA dataset
            domain: 'Urban' or 'Rural'
            split: 'Train' or 'Val'
            max_iters: maximum iterations for training
            augmentations: data augmentation transforms
            img_size: target image size (H, W)
            mean: mean values for normalization
            scale: whether to apply scaling
            mirror: whether to apply mirroring
            ignore_label: ignore label for loss computation
        """
        self.root = root
        self.domain = domain
        self.split = split
        self.img_size = img_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = np.array(mean)
        self.is_mirror = mirror
        self.augmentations = augmentations
        
        # LoveDA class names
        self.class_names = [
            'Background',
            'Building', 
            'Road',
            'Water',
            'Barren',
            'Forest',
            'Agricultural'
        ]
        
        self.n_classes = 7
        
        # Paths to images and masks
        self.images_dir = osp.join(root, split, domain, 'images_png')
        self.masks_dir = osp.join(root, split, domain, 'masks_png')
        
        # Get all image files
        self.img_files = []
        if osp.exists(self.images_dir):
            for fname in os.listdir(self.images_dir):
                if fname.endswith('.png'):
                    img_path = osp.join(self.images_dir, fname)
                    mask_path = osp.join(self.masks_dir, fname)
                    if osp.exists(mask_path):
                        self.img_files.append({
                            'img': img_path,
                            'mask': mask_path,
                            'name': fname
                        })
        
        if not self.img_files:
            raise Exception(f"No files found in {self.images_dir}")
            
        # Repeat dataset for max_iters if specified
        if max_iters is not None:
            self.img_files = self.img_files * int(np.ceil(float(max_iters) / len(self.img_files)))
            
        print(f"Found {len(self.img_files)} {domain} {split} images")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        Reference from https://github.com/Junjue-Wang/LoveDA/blob/master/Unsupervised_Domian_Adaptation/data/loveda.py
        """
        datafiles = self.img_files[index]
        
        # Load image and mask
        image = imread(datafiles["img"])
        mask = imread(datafiles["mask"]).astype(np.int32) - 1

        name = datafiles["name"]
        
        # Resize
        image = np.resize(image, (self.img_size[0], self.img_size[0], 3))
        mask = np.resize(mask, self.img_size)
        
        # Apply augmentations if provided
        if self.augmentations is not None:
            image, mask = self.augmentations(image, mask)
            
        # Convert to float
        image = np.asarray(image, np.float32)
        
        # Store original size
        size = image.shape
        
        # Preprocessing
        image = image[:, :, ::-1]  # RGB to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))  # HWC to CHW
        
        return image.copy(), mask.copy(), np.array(size), name


class LoveDADomainDataSet(data.Dataset):
    """
    LoveDA Dataset for Domain Adaptation Training
    Combines source and target domains
    """
    
    def __init__(self, root, source_domain='Urban', target_domain='Rural', 
                 split='Train', max_iters=None, augmentations=None, 
                 img_size=(512, 512), mean=(123.675, 116.28, 103.53),
                 scale=True, mirror=True, ignore_label=255):
        """
        Args:
            root: path to LoveDA dataset
            source_domain: source domain ('Urban' or 'Rural')
            target_domain: target domain ('Rural' or 'Urban') 
            split: 'Train' or 'Val'
            Other args same as LoveDADataSet
        """
        
        self.source_dataset = LoveDADataSet(
            root=root, domain=source_domain, split=split,
            max_iters=max_iters, augmentations=augmentations,
            img_size=img_size, mean=mean, scale=scale, 
            mirror=mirror, ignore_label=ignore_label
        )
        
        self.target_dataset = LoveDADataSet(
            root=root, domain=target_domain, split=split,
            max_iters=max_iters, augmentations=augmentations,
            img_size=img_size, mean=mean, scale=scale,
            mirror=mirror, ignore_label=ignore_label
        )
        
        self.source_domain = source_domain
        self.target_domain = target_domain
        
    def __len__(self):
        return max(len(self.source_dataset), len(self.target_dataset))
    
    def __getitem__(self, index):
        # Get source sample
        source_idx = index % len(self.source_dataset)
        source_img, source_mask, source_size, source_name = self.source_dataset[source_idx]
        
        # Get target sample  
        target_idx = index % len(self.target_dataset)
        target_img, target_mask, target_size, target_name = self.target_dataset[target_idx]
        
        return {
            'source': (source_img, source_mask, source_size, source_name),
            'target': (target_img, target_mask, target_size, target_name)
        }


# Color palette for visualization
LOVEDA_PALETTE = [
    [255, 255, 255],  # Background - White
    [255, 0, 0],      # Building - Red
    [255, 255, 0],    # Road - Yellow  
    [0, 0, 255],      # Water - Blue
    [159, 129, 183],  # Barren - Purple
    [0, 255, 0],      # Forest - Green
    [255, 195, 128],  # Agricultural - Orange
]


def decode_loveda_mask(mask):
    """Convert mask to colored image for visualization"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(LOVEDA_PALETTE):
        rgb[mask == class_id] = color
        
    return rgb


if __name__ == '__main__':
    # Test the dataset loader
    root_path = "/home/Hung_Data/HungData/mmseg_data/Datasets/LoveDA/loveDA"  # Update this path
    
    # Test single domain dataset
    dataset = LoveDADataSet(root=root_path, domain='Urban', split='Train')
    print(f"Dataset size: {len(dataset)}")
    
    # Test domain adaptation dataset
    da_dataset = LoveDADomainDataSet(
        root=root_path, 
        source_domain='Urban', 
        target_domain='Rural',
        split='Train'
    )
    print(f"Domain adaptation dataset size: {len(da_dataset)}")
    
    # Visualize a sample
    sample = da_dataset[0]
    source_img, source_mask, _, source_name = sample['source']
    target_img, target_mask, _, target_name = sample['target']
    
    print(f"Source: {source_name}, Target: {target_name}")
    print(f"Source image shape: {source_mask.shape}")
    print(f"Target image shape: {source_mask.shape}")