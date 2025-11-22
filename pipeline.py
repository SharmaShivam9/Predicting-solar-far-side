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
        
        # Folder selection logic
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
        if len(parts) > 1: unique_id = '_'.join(parts[1:]) 
        else: unique_id = name_without_ext 

        target_filename = f"HMI_{unique_id}.{self.target_format}"
        parent_dir = os.path.dirname(os.path.dirname(input_path)) 
        return join(parent_dir, 'target', target_filename)

    def __getitem__(self, index):
        # 1. Setup Paths
        label_path = self.label_path_list[index]
        target_path = self.__get_target_path(label_path)
        
        # 2. Define Augmentation Parameters (Calculate once to apply to both Input & Target)
        angle = 0
        pad = 0
        i, j, h, w = 0, 0, 1024, 1024 # Default crop params
        
        if self.opt.is_train:
            angle = randint(-self.opt.max_rotation_angle, self.opt.max_rotation_angle)
            pad = self.opt.padding_size
            # Calculate random crop coordinates
            # Assuming image size increases by 2*pad after padding
            # If original is 1024, padded is 1024 + 2*pad. 
            # We want to crop back to 1024.
            max_offset = 2 * pad if pad > 0 else 0
            i = randint(0, max_offset) # Top
            j = randint(0, max_offset) # Left

        # [ Helper to Load & Process ] -------------------------------------------------
        def process_image(path, format, is_mask=False):
            # A. FAST READ
            if format in ["fits", "fts"]:
                # memmap=False loads directly to RAM, faster for training
                # astype(float32) handles endianness conversion automatically
                arr = fits.getdata(path, memmap=False).astype(np.float32)
            elif format == "npy":
                arr = np.load(path, allow_pickle=True).astype(np.float32)
            else:
                raise NotImplementedError(f"Unknown format: {format}")

            # Handle NaNs
            arr = np.nan_to_num(arr, nan=0.0)

            # B. TO TENSOR (Instant conversion)
            tensor = torch.from_numpy(arr)
            
            # Ensure (C, H, W)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)

            # C. AUGMENTATION (Native Tensor Operations)
            # 1. Rotate
            if angle != 0:
                # Bilinear is smoother for data; Nearest is better if you had discrete classes (but you use floats)
                tensor = TF.rotate(tensor, angle, interpolation=InterpolationMode.BILINEAR)
            
            # 2. Pad
            if pad > 0:
                tensor = TF.pad(tensor, pad, fill=0)
            
            # 3. Crop
            if self.opt.is_train:
                # crop(tensor, top, left, height, width)
                tensor = TF.crop(tensor, i, j, h, w)
            
            return tensor

        # [ Execute ] ------------------------------------------------------------------
        input_tensor = process_image(label_path, self.input_format)
        target_tensor = process_image(target_path, self.target_format)

        # [ Return ] -------------------------------------------------------------------
        input_name = splitext(split(label_path)[-1])[0]
        target_name = splitext(split(target_path)[-1])[0]
        
        return input_tensor, target_tensor, input_name, target_name

    def __len__(self):
        return len(self.label_path_list)