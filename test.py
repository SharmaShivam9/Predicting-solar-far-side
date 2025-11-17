if __name__ == '__main__':
    import os
    import torch
    import numpy as np
    from options import TestOption
    from pipeline import CustomDataset
    from networks import Generator
    from utils import Manager
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from astropy.io import fits
    import torch.nn as nn

    torch.backends.cudnn.benchmark = True
    
    opt = TestOption().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if n_gpus > 0 else "cpu")
    dtype = torch.float32
    STD = opt.dataset_name

    dataset = CustomDataset(opt)
    test_data_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)
    iters = opt.iteration
    step = opt.save_freq
    
    #####################################################################################
    Max_iter = 400000 ######### You can change the Maximum iteration value. #############
    #####################################################################################
    
    # CONSOLIDATED INFERENCE FUNCTION
    def run_inference(iteration_step, opt, device, dtype):
        
        # 1. Determine Model Path
        if iteration_step == -1:
            # Use the 'latest' checkpoint from the new system (saved by train.py/Manager)
            path_model = os.path.join(opt.model_dir, 'latest_dict.pt')
        else:
            # Look for the specific step model
            path_model = os.path.join(opt.model_dir, f'{iteration_step}_dict.pt')

        # Use the correct model directory structure
        dir_image_save = os.path.join(opt.model_dir, '..', 'Image', 'Test', str(iteration_step))
        os.makedirs(dir_image_save, exist_ok=True)
    
        if not os.path.isfile(path_model):
            print(f"Skipping iteration {iteration_step}: Model not found at {path_model}")
            return

        # 2. Load Model State
        # Load the full dictionary, mapping to the current device
        # CHANGED: Load using map_location=device and expecting the dictionary structure
        pt_file = torch.load(path_model, map_location=device) 
        
        G = Generator(opt)
        if torch.cuda.device_count() > 1:
            print("Using multi-GPU for inference")
            G = nn.DataParallel(G)

        G = G.to(device, dtype=dtype)
        G.load_state_dict(pt_file['G_state_dict']) 
        
        manager = Manager(opt)
        # DataLoader initialization moved inside the function to ensure data type consistency
        dataset = CustomDataset(opt)
        test_data_loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.n_workers, shuffle=False)
        
        print(f"Running inference for step: {iteration_step}")

        # 3. Inference Loop (Standardized with torch.no_grad())
        with torch.no_grad():
            G.eval()
            for input, target, input_name, target_name in tqdm(test_data_loader):
                # Ensure input tensor is on the correct device and dtype
                input = input.to(device, dtype=dtype)
                fake = G(input)
                
                UpIB = opt.saturation_upper_limit_target
                LoIB = opt.saturation_lower_limit_target
                    
                # Denormalization 
                np_fake = fake.cpu().numpy().squeeze() *((UpIB - LoIB)/2) +(UpIB+ LoIB)/2
                
                # 4. Saving Results (Using opt.data_format_target for file extension)
                if opt.data_format_target in ["fits", "fts"]:       
                    # <-- CHANGED: Use target_name[0] for the filename
                    fits.writeto(os.path.join(dir_image_save, target_name[0] + '_AI.fits'), np_fake, overwrite=True) 
                elif opt.data_format_target in ["npy"]:
                    # <-- CHANGED: Use target_name[0] for the filename
                    np.save(os.path.join(dir_image_save, target_name[0] + '_AI.npy'), np_fake, allow_pickle=True) 
                else:
                    NotImplementedError("Please check data_format_target option. It has to be fits or npy.")

        print(f"Results saved to: {dir_image_save}")

    # Main execution flow
    iters = opt.iteration
    
    if (iters == -1) and ('latest_dict.pt' in os.listdir(opt.model_dir)):
        # Case 1: Load the single 'latest' checkpoint if it exists
        run_inference(-1, opt, device, dtype)
    elif iters == -1:
        # Case 2: Loop over steps (testing all historical checkpoints)
        step = opt.save_freq 
        for i in range(step, Max_iter + step, step):
            run_inference(i, opt, device, dtype)
    else:
        # Case 3: Run inference for a specific iteration
        run_inference(iters, opt, device, dtype)