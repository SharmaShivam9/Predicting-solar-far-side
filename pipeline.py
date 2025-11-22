import os
from astropy.io import fits
from os.path import split, splitext, join, exists
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate
from random import randint


class CustomDataset(Dataset):
    def __init__(self, opt,mode=None):
        super(CustomDataset, self).__init__()
        self.opt = opt
        dataset_dir = os.path.join(opt.data_root, opt.dataset_name)
        self.input_format = opt.data_format_input
        self.target_format = opt.data_format_target

        if mode is None:
            mode = 'Train' if opt.is_train else 'Test'
        
        if mode == 'Train':
            folder_name = 'Train'
        elif mode == 'Test':
            folder_name = 'Test'
        elif mode == 'Track_Train':
            folder_name = 'Track_Train'
        elif mode == 'Track_Test':
            folder_name = 'Track_Test'
        else:
            raise ValueError(f"Unknown dataset mode: {mode}")

        self.label_path_list = sorted(glob(os.path.join(dataset_dir, folder_name, 'input', '*.' + self.input_format)))


    def __get_target_path(self, input_path):
        input_filename = split(input_path)[-1]
        name_without_ext = splitext(input_filename)[0]
        parts = name_without_ext.split('_')
        if len(parts) > 1:
            unique_id = '_'.join(parts[1:]) 
        else:
            unique_id = name_without_ext 

        # 3. Construct the target filename (e.g., HMI_2011-01-01_00Z.fits)
        target_filename = f"HMI_{unique_id}.{self.target_format}"

        # 4. Construct the full target path.
        # Find the parent folder of 'input' (which is 'Train' or 'Test')
        parent_dir = os.path.dirname(os.path.dirname(input_path)) 
        
        # Join with the 'target' subfolder
        target_path = join(parent_dir, 'target', target_filename)
        
        return target_path

    def __getitem__(self, index):
        
        # [ Training data ] ==============================================================================================
        if self.opt.is_train:
            # Get the path for the input image
            label_path = self.label_path_list[index]
            
            # Use dynamic method to find the corresponding target path
            target_path = self.__get_target_path(label_path)
            
            # --- Augmentation setup (Restored) ---
            self.angle = randint(-self.opt.max_rotation_angle, self.opt.max_rotation_angle)

            self.offset_x = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0
            self.offset_y = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0
            
            # [ Input ] ==================================================================================================
            if self.input_format in ["fits", "fts"]:
                IMG_A0 = np.array(fits.open(label_path)[0].data) # Removed .transpose(2, 0 ,1)
            elif self.input_format in ["npy"]:
                IMG_A0 = np.load(label_path, allow_pickle=True) # Removed .transpose(2, 0 ,1)
            else:
                raise NotImplementedError("Please check data_format_input option. It has to be fits or npy.")
                
            IMG_A0[np.isnan(IMG_A0)] = 0
            
            label_array = self.__rotate(IMG_A0)
            label_array = self.__pad(label_array, self.opt.padding_size)
            label_array = self.__random_crop(label_array)
            label_array = np.ascontiguousarray(label_array, dtype=np.float32)
            
            label_tensor = torch.tensor(label_array, dtype=torch.float32)

            if len(label_tensor.shape) == 2:
                label_tensor = label_tensor.unsqueeze(dim=0)
            
                
            # [ Target ] ==================================================================================================
            
            if self.target_format in ["fits", "fts"]:
                IMG_B0 = np.array(fits.open(target_path)[0].data)
            elif self.target_format in ["npy"]:
                IMG_B0 = np.load(target_path, allow_pickle=True)
            else:
                raise NotImplementedError("Please check data_format_target option. It has to be fits or npy.")
                
            IMG_B0[np.isnan(IMG_B0)] = 0
            
            target_array = self.__rotate(IMG_B0)
            target_array = self.__pad(target_array, self.opt.padding_size)
            target_array = self.__random_crop(target_array)
            target_array = np.ascontiguousarray(target_array, dtype=np.float32)
            
            target_tensor = torch.tensor(target_array, dtype=torch.float32)

            if len(target_tensor.shape) == 2:
                target_tensor = target_tensor.unsqueeze(dim=0)  # Add channel dimension.
            
            
        # [ Test data ] ===================================================================================================
        else:
            # Get paths
            label_path = self.label_path_list[index]
            target_path = self.__get_target_path(label_path) # <--- ADDED: Must get target path for validation
            
            # [ Input ] ==================================================================================================
            # Load and process Input
            if self.input_format in ["fits", "fts"]:                    
                IMG_A0 = np.array(fits.open(label_path)[0].data)
            elif self.input_format in ["npy"]:
                IMG_A0 = np.load(label_path, allow_pickle=True)
            else:
                raise NotImplementedError("Please check data_format_input option. It has to be fits or npy.")
                
            IMG_A0[np.isnan(IMG_A0)] = 0
            IMG_A0 = np.ascontiguousarray(IMG_A0, dtype=np.float32) # From previous fix
            label_tensor = torch.tensor(IMG_A0, dtype=torch.float32)
            
            if len(label_tensor.shape) == 2:
                label_tensor = label_tensor.unsqueeze(dim=0)
                
            # [ Target (for Validation/Real comparison) ] ================================================================
            # <--- ADDED: Load and process Target
            if self.target_format in ["fits", "fts"]:
                IMG_B0 = np.array(fits.open(target_path)[0].data)
            elif self.target_format in ["npy"]:
                IMG_B0 = np.load(target_path, allow_pickle=True)
            else:
                raise NotImplementedError("Please check data_format_target option. It has to be fits or npy.")

            IMG_B0[np.isnan(IMG_B0)] = 0
            IMG_B0 = np.ascontiguousarray(IMG_B0, dtype=np.float32) # From previous fix
            
            target_tensor = torch.tensor(IMG_B0, dtype=torch.float32)

            if len(target_tensor.shape) == 2:
                target_tensor = target_tensor.unsqueeze(dim=0)
            
            # --- RETURN 4 ITEMS FOR VALIDATION LOOP ---
            input_name = splitext(split(label_path)[-1])[0]
            target_name = splitext(split(target_path)[-1])[0]

            # Return the four expected items: input, target, input_name (as '_'), target_name (as 'name')
            return label_tensor, target_tensor, input_name, target_name
        
        # For train, return input, target, and their filenames
        return label_tensor, target_tensor, splitext(split(label_path)[-1])[0], \
                   splitext(split(target_path)[-1])[0]
                   
    def __random_crop(self, x):
        x = np.array(x)
        # Assuming data dimensions are (H, W) or (C, H, W).
        # We crop the last two dimensions.
        if len(x.shape) == 3:
            # Cropping (C, H, W)
            x = x[:, self.offset_x: self.offset_x + 1024, self.offset_y: self.offset_y + 1024]
        else:
            # Cropping (H, W)
            x = x[self.offset_x: self.offset_x + 1024, self.offset_y: self.offset_y + 1024]
            
        return x

    @staticmethod
    def __pad(x, padding_size):
        if type(padding_size) == int:
            if len(x.shape) == 3:
                # Padding (C, H, W) only pads H and W dimensions
                padding_size= ((0, 0), (padding_size, padding_size), (padding_size, padding_size))
            else:
                # Padding (H, W)
                padding_size = ((padding_size, padding_size), (padding_size, padding_size))
        return np.pad(x, pad_width=padding_size, mode="constant", constant_values=0)

    def __rotate(self, x):
        # If the array is 3D (C, H, W), rotate axes 1 and 2 (H and W).
        if len(x.shape) == 3:
            return rotate(x, self.angle, axes=(1, 2), reshape=False)
        # If 2D (H, W), rotate default axes.
        return rotate(x, self.angle, reshape=False)

    @staticmethod
    def __to_numpy(x):
        return np.array(x, dtype=np.float32)

    def __len__(self):
        return len(self.label_path_list)
