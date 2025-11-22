import os
from functools import partial
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.distributed as dist


def get_grid(input, is_real=True):
    # Create tensor ON THE SAME DEVICE as the input
    device = input.device
    if is_real:
        grid = torch.ones_like(input, device=device)
    else:
        grid = torch.zeros_like(input, device=device)
    return grid



def get_norm_layer(type):
    if type == 'BatchNorm2d':
        layer = partial(nn.BatchNorm2d, affine=True)

    elif type == 'InstanceNorm2d':
        layer = partial(nn.InstanceNorm2d, affine=False)

    return layer


def get_pad_layer(type):
    if type == 'reflection':
        layer = nn.ReflectionPad2d

    elif type == 'replication':
        layer = nn.ReplicationPad2d

    elif type == 'zero':
        layer = nn.ZeroPad2d

    else:
        raise NotImplementedError("Padding type {} is not valid."
                                  " Please choose among ['reflection', 'replication', 'zero']".format(type))

    return layer


class Manager(object):
    def __init__(self, opt,current_step=0):
        self.opt = opt
        self.log_path = os.path.join(self.opt.model_dir, 'loss_log.csv')
        self.loss_keys = ['Epoch', 'current_step', 'D_loss', 'G_loss',
                          'L_adv', 'L_fm_weighted', 'L_pil_total', 'L_rec',
                          'L_bg', 'L_nz']
        
        file_exists = os.path.exists(self.log_path)
        is_resuming = current_step > 0

        if not file_exists or not is_resuming:
            print(f"Creating or overwriting log file: {self.log_path}")
            with open(self.log_path, 'w') as f:
                f.write(','.join(self.loss_keys) + '\n')
        else:
            print(f"Resuming from step {current_step}. Checking log file for stale entries...")
            try:
                with open(self.log_path, 'r+') as f:
                    try:
                        header = f.readline()
                        if not header.strip(): raise Exception("Log file is empty.")
                        step_col_index = header.strip().split(',').index('current_step')
                    except (ValueError, Exception):
                        raise Exception("Log file header is corrupted or missing 'current_step'.")

                    last_good_position = f.tell()

                    for line in f:
                        try:
                            line_step = float(line.strip().split(',')[step_col_index])
                        except (IndexError, ValueError, TypeError):
                            break 
                        
                        if line_step <= current_step:
                            last_good_position = f.tell()
                        else:
                            break 
                    
                    f.seek(last_good_position)
                    f.truncate()
                    print("Log file successfully pruned.")

            except Exception as e:
                # If anything goes wrong (e.g., file permissions, corrupt header),
                # fall back to safely recreating the file.
                print(f"Critical error pruning log file ({e}). Recreating log file to prevent corruption.")
                with open(self.log_path, 'w') as f:
                    f.write(','.join(self.loss_keys) + '\n')
    @staticmethod
    def report_loss(package):
        prec = 4 
        
        print("Epoch: {} [{:.{prec}}%] Current_step: {} D_loss: {:.{prec}}  G_loss: {:.{prec}}".
              format(package['Epoch'], package['current_step']/package['total_step'] * 100, package['current_step'],
                     package['D_loss'], package['G_loss'], prec=prec)) # Use prec variable
        
        # 1. NEW: Print Generator breakdown
        if 'L_adv' in package:
            print("  [GEN] Adv: {:.{prec}} | FM (Weighted): {:.{prec}} | Custom Total: {:.{prec}}".
                  format(package['L_adv'], package['L_fm_weighted'], package['L_custom_total'], prec=prec))
        if 'L_rec' in package:
            print("  [Losses] Rec: {:.{prec}} | L_BG: {:.{prec}} | L_NZ: {:.{prec}}".
          format(package['L_rec'], package['L_bg'], package['L_nz'], prec=prec))
        
    
    def log_loss(self, package):
        """Writes all specified loss values to the CSV log file."""
        log_data = []
        for key in self.loss_keys:
            value = package.get(key, np.nan) 
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                # Format floats with 6 decimal places, and integers as is
                if isinstance(value, int):
                    log_data.append(str(value))
                else:
                    log_data.append(f"{value:.6f}")
            else:
                log_data.append('NaN')
        
        with open(self.log_path, 'a') as f:
            f.write(','.join(log_data) + '\n')

    def tensor2image(self, image_tensor):
        """
        Converts a normalized PyTorch tensor (range [-1, 1]) to an 8-bit NumPy image (range [0, 255]).
        This replaces the complex adjust_dynamic_range logic for simplicity.
        """
        np_image = image_tensor.squeeze().cpu().float().numpy()

        if len(np_image.shape) == 3:
            np_image = np.transpose(np_image, (1, 2, 0))  # Convert C, H, W to H, W, C
        
        np_image = (np_image + 1.0) / 2.0 * 255.0
        
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np_image

    def save_image(self, image_tensor, path):
        Image.fromarray(self.tensor2image(image_tensor)).save(path, self.opt.image_mode)

    def save(self, package, image=False, model=False):
        if image:
            path_real = os.path.join(self.opt.image_dir, str(package['Epoch']) + '_' + 'real.png')
            path_fake = os.path.join(self.opt.image_dir, str(package['Epoch']) + '_' + 'fake.png')
            self.save_image(package['target_tensor'], path_real)
            self.save_image(package['generated_tensor'], path_fake)

        elif model:
            path_pt = os.path.join(self.opt.model_dir, str(package['current_step']) + '_' + 'dict.pt')
            torch.save(package, path_pt)
            path_latest = os.path.join(self.opt.model_dir, 'latest_dict.pt')
            torch.save(package, path_latest)
            checkpoint_list = [f for f in os.listdir(self.opt.model_dir) if f.endswith('_dict.pt') and f != 'latest_dict.pt']
            checkpoint_list_sorted = sorted(checkpoint_list,key=lambda x: int(x.split('_')[0]))
            while len(checkpoint_list_sorted) > 5:
                old_ckpt = checkpoint_list_sorted.pop(0)
                try:
                    os.remove(os.path.join(self.opt.model_dir, old_ckpt))
                    print(f"Deleted old checkpoint: {old_ckpt}")
                except Exception as e:
                    print(f"Could not delete {old_ckpt}: {e}")



    def __call__(self, package):
        if package['current_step'] % self.opt.report_freq == 0:
            self.report_loss(package)
            self.log_loss(package)

        if package['current_step'] % self.opt.save_freq == 0:
            # NOTE: synchronization must be done by the training script (train.py)
            # because Manager is only instantiated on rank 0. Do not call dist.barrier()
            # here — that would deadlock.
            self.save(package, model=True)




def weights_init(module):
    
    if isinstance(module, nn.Conv2d):
        module.weight.detach().normal_(0.0, 0.02)

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().normal_(1.0, 0.02)
        module.bias.detach().fill_(0.0)
