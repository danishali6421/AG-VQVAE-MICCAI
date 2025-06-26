import logging
import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from config.configp import get_args
from src.transformations import get_train_transforms, get_val_transforms

import cv2
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing
from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing
import numpy as np
import random
import json



class BrainTumorDataset(Dataset):
    def __init__(self, json_path, crop_size, split='train'):
        self.crop_size = crop_size
        self.split = split
        self.modalities = ["t1c", "t1n", "t2f", "t2w"]

        # Load JSON
        with open(json_path, 'r') as f:
            all_data = json.load(f)
        
        self.data_list = all_data[split]

        if split == 'train':
            self.transforms = get_train_transforms(crop_size)
        else:
            self.transforms = get_val_transforms(crop_size)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        data_dict = {}

        # Assign each modality to its name
        for i, modality in enumerate(self.modalities):
            data_dict[modality] = item['image'][i]

        data_dict['mask'] = item['mask']

        if self.transforms:
            data_dict = self.transforms(data_dict)

        return data_dict



def Dataloading(json_path, crop_size):
    train_dataset = BrainTumorDataset(json_path=json_path, crop_size=crop_size, split='train')
    val_dataset = BrainTumorDataset(json_path=json_path, crop_size=crop_size, split='val')
    test_dataset = BrainTumorDataset(json_path=json_path, crop_size=crop_size, split='test')
    return train_dataset, val_dataset, test_dataset

    

if __name__ == "__main__":
    args = get_args()
    data_path="../dataset/processed/"
    crop_size=args.crop_size
    modalities=args.modalities
    
    print(data_path)
    
    train_dataset, val_dataset, test_dataset=Dataloading(data_path, crop_size, modalities)
    
    # train_dataset = BrainTumorDataset(data_path=args.data_path, modalities=args.modalities, crop_size=args.crop_size, split='train')
    # val_dataset = BrainTumorDataset(data_path=args.data_path, modalities=args.modalities, crop_size=args.crop_size, split='val')
    # test_dataset = BrainTumorDataset(data_path=args.data_path, modalities=args.modalities, crop_size=args.crop_size, split='test')

    # Print out dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Iterate through a few batches and print shapes
    for batch_idx, batch_data in enumerate(train_loader):
        print(f'Training Batch {batch_idx+1}:')
        #print((batch_data).keys())
        print(f'Images batch shape of train modality:', [batch_data[key].shape for key in ['t1', 't2', 't1ce', 'flair']])
        print('Masks train batch shape:', batch_data['mask'].shape)
                
        if batch_idx >= 2:
            break

    for batch_idx, batch_data in enumerate(val_loader):
        print(f'Validation Batch {batch_idx+1}:')
        print(f'Images batch shape of val modality:', [batch_data[key].shape for key in ['t1', 't2', 't1ce', 'flair']])
        print('Masks val batch shape:', batch_data['mask'].shape)
        if batch_idx >= 2:
            break

    for batch_idx, batch_data in enumerate(test_loader):
        print(f'Test Batch {batch_idx+1}:')
        print(f'Images batch shape of test modality:', [batch_data[key].shape for key in ['t1', 't2', 't1ce', 'flair']])
        print('Masks test batch shape:', batch_data['mask'].shape)
        if batch_idx >= 2:
            break