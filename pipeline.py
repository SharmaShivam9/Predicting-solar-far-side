import os
from astropy.io import fits
from os.path import split, splitext, join
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from random import randint
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class CustomDataset(Dataset):
    def __init__(self, opt, mode=None):
        super(CustomDataset, self).__init__()
        self.opt = opt
        dataset_dir = os.path.join(opt.data_root, opt.dataset_name)
        self.input_format = opt.data_format_input
        self.target_format = opt.data_format_target

        if mode is None:
            mode = 'Train' if opt.is_train else 'Test'
        
        # Select folder
        if mode == 'Train': folder_name = 'Train'
        elif mode == 'Test': folder_name = 'Test'
        elif mode == 'Track_Train': folder_name = 'Track_Train'
        elif mode == 'Track_Test': folder_name = 'Track_Test'
        else: raise ValueError(f"Unknown dataset mode: {mode}")

        self.label_path_list = sorted(glob(os.path.join(dataset_dir, folder_name, 'input', '*.' + self.input_format)))

    def __get_target_path(self, input_path):
        input_filename = split(input_path)[-1]
        name_without_ext = splitext(input_filename)[0]
        parts = name_without_ext.split('_')
        unique_id = '_'.join(parts[1:]) if len(parts) > 1 else name_without_ext 
        target_filename = f"HMI_{unique_id}.{self.target_format}"
        parent_dir = os.path.dirname(os.path.dirname(input_path)) 
        return join(parent_dir, 'target', target_filename)

    def __getitem__(self, index):
        # 1. Setup Paths
        label_path = self.label_path_list[index]
        target_path = self.__get_target_path(label_path)
        
        # 2. Define Augmentation Params (Computed once, applied to both)
        angle = 0
        pad = 0
        i, j, h, w = 0, 0, 1024, 1024
        
        if self.opt.is_train:
            angle = randint(-self.opt.max_rotation_angle, self.opt.max_rotation_angle)
            pad = self.opt.padding_size
            if pad > 0:
                # If we pad, we pick a random crop to return to original size
                max_offset = 2 * pad
                i = randint(0, max_offset)
                j = randint(0, max_offset)

        # --- HELPER FUNCTION ---
        def load_and_process(path, format, is_input):
            # A. LOAD TO MEMORY
            if format in ["fits", "fts"]:
                # memmap=False forces a read to RAM (faster for training).
                # astype(float32) handles the Big-Endian to Little-Endian conversion.
                arr = fits.getdata(path, memmap=False).astype(np.float32)
            elif format == "npy":
                arr = np.load(path, allow_pickle=True).astype(np.float32)
            
            arr = np.nan_to_num(arr, nan=0.0)

            # B. TO TENSOR (SAFE COPY)
            # Use torch.tensor() not from_numpy() to ensure contiguous memory and avoid the DDP crash.
            tensor = torch.tensor(arr, dtype=torch.float32)

            # C. SHAPE CORRECTION (PyTorch needs [Channels, H, W])
            # If your fits is saved as [H, W, Channels] (pixel-last), we permute it.
            if tensor.ndim == 3 and tensor.shape[2] <= 4: 
                 tensor = tensor.permute(2, 0, 1) # [1024,1024,4] -> [4,1024,1024]
            
            # If data is [H, W] (2D), make it [1, H, W]
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)

            # D. AUGMENTATION (TF.rotate handles multi-channel [4, H, W] correctly)
            if angle != 0:
                tensor = TF.rotate(tensor, angle, interpolation=InterpolationMode.BILINEAR)
            
            if pad > 0:
                tensor = TF.pad(tensor, pad, fill=0)
                if self.opt.is_train:
                    tensor = TF.crop(tensor, i, j, h, w)
            
            return tensor

        # --- EXECUTE ---
        input_tensor = load_and_process(label_path, self.input_format, is_input=True)
        target_tensor = load_and_process(target_path, self.target_format, is_input=False)

        return input_tensor, target_tensor, splitext(split(label_path)[-1])[0], splitext(split(target_path)[-1])[0]

    def __len__(self):
        return len(self.label_path_list)