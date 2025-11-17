import os
import argparse
import numpy as np

SATURATION_GAUSS = 3000.0
CUSTOM_LAMBDAS = {
    'L_rec': 1,
    'L_bg': 10,
    'L_nz': 10
}
FINAL_THRESHOLDS = {
    'L_bg':0.1,
    'L_nz':10
}
class BaseOption(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--gpu_ids', type=int, default=0, help='gpu number. If -1, use cpu')
        self.parser.add_argument('--data_format_input', type=str, default='fits',
                                 help="Input data extension. This will be used for loading and saving. [fits or npy]")
        self.parser.add_argument('--data_format_target', type=str, default='fits',
                                 help="Target data extension. This will be used for loading and saving. [fits or npy]")

        # data option
        self.parser.add_argument('--input_ch', type=int, default=4, help="# of input channels for Generater")
        self.parser.add_argument('--saturation_lower_limit_target', type=float, default=-3000, help="Saturation value (lower limit) of target")
        self.parser.add_argument('--saturation_upper_limit_target', type=float, default=3000, help="Saturation value (upper limit) of target")

        # data augmentation
        self.parser.add_argument('--batch_size', type=int, default=8, help='the number of batch_size')
        self.parser.add_argument('--data_root', type=str, default='./datasets', help='Root directory where dataset subfolders are located')
        self.parser.add_argument('--dataset_name', type=str, default='AIA_to_HMI', help='dataset directory name')
        self.parser.add_argument('--data_type', type=int, default=32, help='float dtype')
        self.parser.add_argument('--image_mode', type=str, default='png', help='extension for saving image')
        self.parser.add_argument('--n_downsample', type=int, default=4, help='how many times you want to downsample input data in G')
        self.parser.add_argument('--n_residual', type=int, default=9, help='the number of residual blocks in G')
        self.parser.add_argument('--n_workers', type=int, default=16, help='how many threads you want to use')
        self.parser.add_argument('--norm_type', type=str, default='InstanceNorm2d', help='[BatchNorm2d, InstanceNorm2d]')
        self.parser.add_argument('--padding_type', type=str, default='reflection', help='[reflection, replication, zero]')
        self.parser.add_argument('--padding_size', type=int, default=0, help='padding size')
        self.parser.add_argument('--max_rotation_angle', type=int, default=0, help='rotation angle in degrees')
        self.parser.add_argument('--val_during_train', action='store_true', default=True)

        self.parser.add_argument('--report_freq', type=int, default=20)
        self.parser.add_argument('--save_freq', type=int, default=2000)
        self.parser.add_argument('--display_freq', type=int, default=100)
        
    def parse(self):
        opt, _ = self.parser.parse_known_args() 
        opt.format = 'png'
        opt.n_df = 64
        opt.flip = False

        opt.n_gf = 32
        opt.output_ch = 1

        if opt.data_type == 16:
            opt.eps = 1e-4
        elif opt.data_type == 32:
            opt.eps = 1e-8 

        global SATURATION_GAUSS, CUSTOM_LAMBDAS, FINAL_THRESHOLDS
        
        opt.SATURATION_GAUSS = SATURATION_GAUSS
        opt.CUSTOM_LAMBDAS = CUSTOM_LAMBDAS
        opt.FINAL_THRESHOLDS = FINAL_THRESHOLDS
        # ---------------------------------------------
        dataset_name = opt.dataset_name

        # --- 1. Define Base Paths ---
        base_checkpoint_dir = os.path.join('./checkpoints', dataset_name)
        opt.model_dir = os.path.join(base_checkpoint_dir, 'Model')
        base_image_dir = os.path.join(base_checkpoint_dir, 'Image')

        # --- 2. Define and Create New Image Directories ---
        opt.track_train_dir = os.path.join(base_image_dir, 'Track_Train')
        opt.track_test_dir = os.path.join(base_image_dir, 'Track_Test')
        opt.full_test_dir = os.path.join(base_image_dir, 'Full_Test')


        # Create all directories
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.track_train_dir, exist_ok=True)
        os.makedirs(opt.track_test_dir, exist_ok=True)
        os.makedirs(opt.full_test_dir, exist_ok=True)
        

        
        opt.image_dir = opt.full_test_dir

        return opt


class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument('--is_train', type=bool, default=True, help='train flag')
        self.parser.add_argument('--n_epochs', type=int, default=600, help='how many epochs you want to train')
        self.parser.add_argument('--latest', type=int, default=0, help='Resume epoch')

        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--epoch_decay', type=int, default=100, help='when to start decay the lr')
        self.parser.add_argument('--GAN_type', type=str, default='LSGAN', help='[GAN, LSGAN, WGAN_GP]')
        self.parser.add_argument('--lambda_FM', type=int, default=20, help='weight for FM loss')
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--n_D', type=int, default=2, help='how many discriminators in differet scales you want to use')      
        self.parser.add_argument('--no_shuffle', action='store_true', default=False, help='if you want to shuffle the order')
        


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()
        
        self.parser.add_argument('--is_train', type=bool, default=False, help='test flag')
        self.parser.add_argument('--iteration', type=int, default=-1, help='if you want to generate from input for the specific iteration')
        self.parser.add_argument('--no_shuffle', type=bool, default=True, help='if you want to shuffle the order')
