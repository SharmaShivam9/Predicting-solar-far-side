import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io.fits import getdata
import torch.nn.functional as F
from random import randint
from os.path import split, splitext, join


class CustomDataset(Dataset):
    def __init__(self, opt, mode=None):
        super().__init__()
        self.opt = opt
        dataset_dir = os.path.join(opt.data_root, opt.dataset_name)

        if mode is None:
            mode = "Train" if opt.is_train else "Test"

        if mode not in ["Train", "Test", "Track_Train", "Track_Test"]:
            raise ValueError(f"Invalid dataset mode: {mode}")

        folder = mode
        self.label_path_list = sorted(
            glob(os.path.join(dataset_dir, folder, "input", "*.fits"))
        )

        self.is_train = opt.is_train
        self.padding = opt.padding_size
        self.max_angle = opt.max_rotation_angle


    # ---------- Target path resolver ----------
    def __get_target_path(self, input_path):
        file = split(input_path)[-1]
        base = splitext(file)[0]       # e.g. AIA_2011-01-01_00Z
        parts = base.split("_")
        uid = "_".join(parts[1:]) if len(parts) > 1 else base
        target_name = f"HMI_{uid}.fits"

        parent = os.path.dirname(os.path.dirname(input_path))
        return join(parent, "target", target_name)


    # ---------- FAST GPU rotation ----------
    def _rotate(self, tensor, angle_deg):
        if angle_deg == 0:
            return tensor
        angle = torch.tensor([angle_deg * np.pi / 180], dtype=tensor.dtype, device=tensor.device)
        theta = torch.zeros((1, 2, 3), dtype=tensor.dtype, device=tensor.device)
        theta[:, 0, 0] = torch.cos(angle)
        theta[:, 0, 1] = -torch.sin(angle)
        theta[:, 1, 0] = torch.sin(angle)
        theta[:, 1, 1] = torch.cos(angle)

        N, C, H, W = 1, 1, tensor.shape[-2], tensor.shape[-1]
        grid = F.affine_grid(theta, size=(N, C, H, W), align_corners=False)
        return F.grid_sample(tensor.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros",
                             align_corners=False).squeeze(0)


    # ---------- FAST crop (no pad needed) ----------
    def _crop(self, x, offset_x, offset_y):
        return x[..., offset_x:offset_x+1024, offset_y:offset_y+1024]


    def __getitem__(self, index):

        label_path = self.label_path_list[index]
        target_path = self.__get_target_path(label_path)

        # ---------------- LOAD FITS (fast, memmap) ----------------
        IMG_A = getdata(label_path, memmap=True).astype(np.float32)
        IMG_A[np.isnan(IMG_A)] = 0

        IMG_B = getdata(target_path, memmap=True).astype(np.float32)
        IMG_B[np.isnan(IMG_B)] = 0

        # Channel add
        A = torch.from_numpy(IMG_A).unsqueeze(0)
        B = torch.from_numpy(IMG_B).unsqueeze(0)

        if self.is_train:

            # ---------------- AUGMENTATION: local values ----------------
            angle = randint(-self.max_angle, self.max_angle)
            offset_x = randint(0, 2 * self.padding - 1) if self.padding > 0 else 0
            offset_y = randint(0, 2 * self.padding - 1) if self.padding > 0 else 0

            # ---------------- FAST GPU ROTATE ----------------
            A = self._rotate(A, angle)
            B = self._rotate(B, angle)

            # ---------------- CROP ----------------
            A = self._crop(A, offset_x, offset_y)
            B = self._crop(B, offset_x, offset_y)

            return A, B, splitext(os.path.basename(label_path))[0], \
                   splitext(os.path.basename(target_path))[0]

        else:
            # -------- Validation/Test: no augmentation --------
            return (
                A,
                B,
                splitext(os.path.basename(label_path))[0],
                splitext(os.path.basename(target_path))[0],
            )


    def __len__(self):
        return len(self.label_path_list)
