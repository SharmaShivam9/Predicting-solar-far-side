import torch
import torch.nn as nn
import numpy as np
from utils import get_grid, get_norm_layer, get_pad_layer


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        act = nn.ReLU(inplace=True)
        input_ch = opt.input_ch
        n_gf = opt.n_gf
        norm = get_norm_layer(opt.norm_type)
        output_ch = opt.output_ch
        pad = get_pad_layer(opt.padding_type)

        model = []
        model += [pad(3), nn.Conv2d(input_ch, n_gf, kernel_size=7, padding=0), norm(n_gf), act]

        for _ in range(opt.n_downsample):
            model += [nn.Conv2d(n_gf, 2 * n_gf, kernel_size=3, padding=1, stride=2), norm(2 * n_gf), act]
            n_gf *= 2

        for _ in range(opt.n_residual):
            model += [ResidualBlock(n_gf, pad, norm, act)]

        for _ in range(opt.n_downsample):
            model += [nn.ConvTranspose2d(n_gf, n_gf//2, kernel_size=3, padding=1, stride=2, output_padding=1),
                      norm(n_gf//2), act]
            n_gf //= 2

        model += [pad(3), nn.Conv2d(n_gf, output_ch, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)

        print(self)
        print("the number of G parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        return self.model(x)
        

class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()

        act = nn.LeakyReLU(0.2, inplace=True)
        input_channel = opt.input_ch + opt.output_ch
        n_df = opt.n_df
        norm = nn.InstanceNorm2d

        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2), norm(2 * n_df), act]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2), norm(4 * n_df), act]]
        blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1), norm(8 * n_df), act]]


        blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            result.append(block(result[-1]))

        return result[1:]  # except for the input


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        
        for i in range(opt.n_D):
            setattr(self, 'Scale_{}'.format(str(i)), PatchDiscriminator(opt))
        self.n_D = 2

        print(self)
        print("the number of D parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        result = []
        for i in range(self.n_D):
            result.append(getattr(self, 'Scale_{}'.format(i))(x))
            if i != self.n_D - 1:
                x = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(x)
        return result


class Loss(object):
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:0' if opt.gpu_ids != -1 and torch.cuda.is_available() else 'cpu')

        self.criterion = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.n_D = 2
        
        self.CUSTOM_LAMBDAS = opt.CUSTOM_LAMBDAS
        self.FINAL_THRESHOLDS = opt.FINAL_THRESHOLDS
        self.SATURATION_GAUSS = opt.SATURATION_GAUSS
        

    def __call__(self, D, G, input, target):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        fake = G(input)

        # ----------------------------------------------------
        # 1. DISCRIMINATOR LOSS (D_loss) - ADVERSARIAL ONLY
        # ----------------------------------------------------
        real_features = D(torch.cat((input, target), dim=1))
        fake_features_D = D(torch.cat((input, fake.detach()), dim=1)) # Renamed to avoid confusion

        # *** FIX: Create a detached copy of the real features for FM loss. ***
        # This prevents D_loss.backward() from consuming the graph needed later.
        real_features_for_fm = [[f.detach() for f in scale] for scale in real_features]

        for i in range(self.n_D):
            # D_loss uses the ORIGINAL (non-detached) real features
            real_grid = get_grid(real_features[i][-1], is_real=True).to(self.device)
            fake_grid = get_grid(fake_features_D[i][-1], is_real=False).to(self.device)

            # MSE Loss for D: Real should be 1, Fake should be 0
            loss_D += (self.criterion(real_features[i][-1], real_grid) +
                       self.criterion(fake_features_D[i][-1], fake_grid)) * 0.5
        
        
        fake_features_G = D(torch.cat((input, fake), dim=1)) # Rerun D on non-detached fake

        # --- A. ADVERSARIAL LOSS (G should fool D -> target 1.0) ---
        loss_G_adv = 0.0 # Use loss_G_adv instead of loss_G
        for i in range(self.n_D):
            real_grid = get_grid(fake_features_G[i][-1], is_real=True).to(self.device)
            loss_G_adv += self.criterion(fake_features_G[i][-1], real_grid)
            
            # --- B. FEATURE MATCHING (FM) LOSS ---
            for j in range(len(fake_features_G[i]) - 1): # Exclude the last output layer
                # FIX: Use the DETACHED copy of REAL features for the comparison
                loss_G_FM += self.FMcriterion(fake_features_G[i][j], real_features_for_fm[i][j])
                
        loss_G_adv_fm = loss_G_adv + loss_G_FM * (1.0 / self.opt.n_D) * self.opt.lambda_FM
        
        L_custom, custom_losses = self._calculate_custom_loss(input, target, fake)

        # Total Generator Loss
        L_total_G = loss_G_adv_fm + L_custom

        custom_losses['L_adv'] = loss_G_adv.detach().item()
        custom_losses['L_fm_weighted'] = (loss_G_FM * (1.0 / self.opt.n_D) * self.opt.lambda_FM).detach().item()
        # The loss components returned are D_loss, Total G_loss, target, and fake.
        return loss_D, L_total_G, target, fake, custom_losses
    

    def _calculate_custom_loss(self, input, target, generated):
        N, C, H, W = target.shape

		# --- UN-NORMALIZE to Physical Units (Gauss) on float32 tensors ---
        UpIB = self.opt.saturation_upper_limit_target
        LoIB = self.opt.saturation_lower_limit_target
		
        Range = UpIB - LoIB
        Offset = (UpIB + LoIB) / 2.0
		
        B_gen_gauss = generated * (Range / 2.0) + Offset
        B_target_gauss = target * (Range / 2.0) + Offset 

        mask = input[:, 3:4, :, :]
      
        mask_squeezed = mask.squeeze(1)
        
        sun_pixel_count = mask.sum() + 1e-8 # 
        L_rec = (torch.abs(generated- target) * mask).sum() / sun_pixel_count
        L_rec_loss = self.CUSTOM_LAMBDAS.get('L_rec', 0.0) * L_rec
		
		# --- Prepare data for Custom calculations ---
        B_gen = B_gen_gauss.squeeze(1) # [N, H, W]
        B_norm_gen = B_gen / self.opt.SATURATION_GAUSS 
		
        # --- 3. ACTIVE ANTI-COLLAPSE PENALTIES ---
		
		# L_bg: Geometric Background Enforcement Loss 
        background_mask = 1.0 - mask # Shape [N, 1, H, W]
        background_pixel_count = background_mask.sum()
        NEAR_ZERO_BG_THRESHOLD = self.opt.FINAL_THRESHOLDS['L_bg']
        violating_bg_pixels_mask = (torch.abs(generated) > NEAR_ZERO_BG_THRESHOLD) & background_mask.bool()
        B_norm_gen_magnitude = torch.abs(B_norm_gen)
        L_bg_loss = B_norm_gen_magnitude[violating_bg_pixels_mask.squeeze(1)].sum() / background_pixel_count
        L_bg_penalty = self.CUSTOM_LAMBDAS.get('L_bg', 0.0) * L_bg_loss

        
		# L_nz: Anti-Zero Enforcement Loss (Correctly uses the dynamic input mask)
        L_nz_lambda = self.CUSTOM_LAMBDAS.get('L_nz', 0.0)
        THRESHOLD_GAUSS = self.opt.FINAL_THRESHOLDS['L_nz'] # Define the threshold in Gauss
        gen_magnitude = torch.abs(B_gen)
        target_magnitude = torch.abs(B_target_gauss.squeeze(1))
        gen_below_thresh = (gen_magnitude < THRESHOLD_GAUSS) & mask_squeezed.bool()
        target_above_thresh = (target_magnitude >= THRESHOLD_GAUSS) & mask_squeezed.bool()
        error_pixels_mask = gen_below_thresh & target_above_thresh
        L_nz_loss = error_pixels_mask.sum() / sun_pixel_count
        L_nz_penalty = L_nz_lambda * L_nz_loss
				  
        L_custom_total= L_rec_loss + L_bg_penalty + L_nz_penalty
		
        custom_losses = {
			'L_rec': L_rec_loss.mean().detach().item(), 
			'L_bg': L_bg_penalty.detach().item(),
			'L_nz': L_nz_penalty.detach().item(),
			'L_custom_total': L_custom_total.mean().detach().item(), 
		}

		# Return the float16 version of the total loss
        return L_custom_total, custom_losses
    
class ResidualBlock(nn.Module):
    def __init__(self, n_channels, pad, norm, act):
        super(ResidualBlock, self).__init__()
        block = [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels), act]
        block += [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)
