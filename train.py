if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    from networks import Discriminator, Generator, Loss
    from options import TrainOption
    from pipeline import CustomDataset
    from utils import Manager, weights_init
    from torch.optim.lr_scheduler import LambdaLR
    import numpy as np
    from tqdm import tqdm
    import datetime
    import os
    import glob

    torch.backends.cudnn.benchmark = True

    opt = TrainOption().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    device = torch.device('cuda:0')
    dtype = torch.float16 if opt.data_type == 16 else torch.float32

    if opt.val_during_train:
        from options import TestOption
        test_opt = TestOption().parse()
        save_freq = opt.save_freq
        
        test_dataset = CustomDataset(test_opt, mode='Test')
        test_data_loader = DataLoader(dataset=test_dataset,
                                      batch_size=test_opt.batch_size,
                                      num_workers=opt.n_workers,
                                      shuffle=not test_opt.no_shuffle)
        
        # --- 2. SETUP Track_Train ---
        print("Setting up Track_Train dataloader...")
        track_train_dataset = CustomDataset(test_opt, mode='Track_Train')
        track_train_loader = DataLoader(dataset=track_train_dataset,
                                         batch_size=1, # Use batch size 1 for individual tracking
                                         num_workers=opt.n_workers,
                                         shuffle=False)

        # --- 3. SETUP Track_Test ---
        print("Setting up Track_Test dataloader...")
        track_test_dataset = CustomDataset(test_opt, mode='Track_Test')
        track_test_loader = DataLoader(dataset=track_test_dataset,
                                        batch_size=1, # Use batch size 1
                                        num_workers=opt.n_workers,
                                        shuffle=False)

        print("Pre-saving REAL images for tracking folders...")
        temp_manager = Manager(test_opt) 
        
        # Function to un-normalize and save
        def presave_real_images(loader, base_dir):
            for _, target, _, name in tqdm(loader, desc=f"Pre-saving REALs in {os.path.basename(base_dir)}"):
                img_name = name[0] # Get filename string
                track_image_dir = os.path.join(base_dir, img_name)
                os.makedirs(track_image_dir, exist_ok=True)
                
                real_image_path = os.path.join(track_image_dir, f"_REAL_{img_name}.png")
                
                if not os.path.exists(real_image_path):                    
                    temp_manager.save_image(target, path=real_image_path)
        
        with torch.no_grad():
            presave_real_images(track_train_loader, opt.track_train_dir)
            presave_real_images(track_test_loader, opt.track_test_dir)
        
        print("Tracking setup complete.")
        
    else:
        save_freq = opt.save_freq


    init_lr = opt.lr
    lr = opt.lr

    dataset = CustomDataset(opt)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             num_workers=opt.n_workers,
                             shuffle=not opt.no_shuffle)

    G = Generator(opt).apply(weights_init).to(device=device)
    D = Discriminator(opt).apply(weights_init).to(device=device)

    criterion = Loss(opt)

    G_optim = torch.optim.Adam(G.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=1e-4)
    D_optim = torch.optim.Adam(D.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=1e-4)

    lambda_rule = lambda epoch: 1.0 - max(0, epoch - opt.epoch_decay) / float(opt.n_epochs - opt.epoch_decay + 1)
    G_scheduler = LambdaLR(G_optim, lr_lambda=lambda_rule)
    D_scheduler = LambdaLR(D_optim, lr_lambda=lambda_rule)

    # --- Robust Resume Logic ---
    latest_path = os.path.join(opt.model_dir, 'latest_dict.pt')
    resume_path = None
    if opt.latest:
        specific_path = os.path.join(opt.model_dir, str(opt.latest) + '_dict.pt')
        if os.path.isfile(specific_path):
            resume_path = specific_path
    
    if resume_path is None and os.path.isfile(latest_path):
        resume_path = latest_path

    if resume_path:
        pt_file = torch.load(resume_path, map_location=device, weights_only=False) 
        
        init_epoch = pt_file['Epoch']
        print(f"Resume at epoch: {init_epoch} from file: {os.path.basename(resume_path)}")
        
        G.load_state_dict(pt_file['G_state_dict'])
        D.load_state_dict(pt_file['D_state_dict'])
        
        G_optim.load_state_dict(pt_file['G_optim_state_dict'])
        D_optim.load_state_dict(pt_file['D_optim_state_dict'])

        # FIX: Temporarily inject a high learning rate to escape local minima
        NEW_WAKEUP_LR = 0.0002
        for param_group in G_optim.param_groups:
            param_group['lr'] = NEW_WAKEUP_LR
        for param_group in D_optim.param_groups:
            param_group['lr'] = NEW_WAKEUP_LR
            
        # Resume Schedulers
        G_scheduler.load_state_dict(pt_file['G_scheduler_state_dict'])
        D_scheduler.load_state_dict(pt_file['D_scheduler_state_dict'])
        
        current_step = pt_file['current_step'] 
    else:
        init_epoch = 1
        current_step = 0
    # ----------------------------------------------------------------------------------

    manager = Manager(opt,current_step)
    
    if opt.val_during_train and current_step > 0:
        if current_step % save_freq == 0:
            step_to_validate = current_step
            print(f"\nResumed at step {step_to_validate}. Rerunning validation for crash recovery.")
            
            G.eval()
            test_image_dir = os.path.join(opt.full_test_dir, str(step_to_validate)) 
            os.makedirs(test_image_dir, exist_ok=True)
            
            with torch.no_grad():
                    for input, target, _, name in tqdm(test_data_loader, desc=f"Rerunning Val @ {step_to_validate}"):
                        input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)
                        fake = G(input) 

                        if len(fake.shape) == 2:
                            fake = fake.unsqueeze(0)
                        if len(target.shape) == 2:
                            target = target.unsqueeze(0)

                        manager.save_image(fake, path=os.path.join(test_image_dir, 'Check_{:d}_'.format(step_to_validate)+ name[0] + '_fake.png'))
                        manager.save_image(target, path=os.path.join(test_image_dir, 'Check_{:d}_'.format(step_to_validate)+ name[0] + '_real.png'))
            
            G.train()
    # --------------------------------------------------------------------------


    total_step = opt.n_epochs * len(data_loader)
    start_time = datetime.datetime.now()
    for epoch in range(init_epoch, opt.n_epochs + 1):
        # FIX: Use enumerate to get the index for gradient accumulation check
        for batch_index,(input, target, _, _) in enumerate(tqdm(data_loader)):
            G.train()
            current_step += 1 
            
            input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)
            
            D_loss, G_loss, target_tensor, generated_tensor, custom_losses = criterion(D, G, input, target)
            
           # --- 1. CALCULATE DISCRIMINATOR GRADIENTS ---
            D_optim.zero_grad()
            D_loss.backward(retain_graph=True) # Keep graph for G_loss
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            
            # --- 2. CALCULATE GENERATOR GRADIENTS ---
            G_optim.zero_grad()
            G_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            
            # --- 3. UPDATE WEIGHTS (NOW THAT ALL GRADS ARE COMPUTED) ---
            D_optim.step()
            G_optim.step()
            
            
            package = {'Epoch': epoch,
                           'current_step': current_step,
                           'total_step': total_step,
                           'D_loss': D_loss.detach().item(), 
                           'G_loss': G_loss.detach().item(),
                           'D_state_dict': D.state_dict(),
                           'G_state_dict': G.state_dict(),
                           'D_optim_state_dict': D_optim.state_dict(),
                           'G_optim_state_dict': G_optim.state_dict(),
                           'G_scheduler_state_dict': G_scheduler.state_dict(),
                           'D_scheduler_state_dict': D_scheduler.state_dict(),
                           'target_tensor': target_tensor,
                           'generated_tensor': generated_tensor.detach()}
                
            package.update(custom_losses)
            manager(package)

            # --- FULL_TEST (Runs every 'save_freq' steps) ---
            if opt.val_during_train and (current_step % save_freq == 0):
                    G.eval()
                    print(f"\n--- Running Full Test for Step {current_step} ---")
                    
                    # The base directory (opt.full_test_dir) is set in options.py
                    step_test_image_dir = os.path.join(opt.full_test_dir, str(current_step))
                    os.makedirs(step_test_image_dir, exist_ok=True)

                    with torch.no_grad():
                        for input, target, _, name in tqdm(test_data_loader, desc=f"Running Full Test @ {current_step}"): 
                            input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)
                            fake = G(input)


                            if len(fake.shape) == 2: fake = fake.unsqueeze(0)
                            if len(target.shape) == 2: target = target.unsqueeze(0)
                            # --- End Un-normalization ---

                            manager.save_image(fake, path=os.path.join(step_test_image_dir, 'Check_{:d}_'.format(current_step)+ name[0] + '_fake.png'))
                            manager.save_image(target, path=os.path.join(step_test_image_dir, 'Check_{:d}_'.format(current_step)+ name[0] + '_real.png'))
                    
                    G.train() 
                    print("--- Full Test Complete ---")

        if opt.val_during_train:
            G.eval()
            print(f"\n--- Running Epoch {epoch} Tracking ---")

            # Helper function for un-normalizing fake images


            with torch.no_grad():
                # 1. RUN Track_Train
                for input, _, _, name in tqdm(track_train_loader, desc=f"Tracking Train @ Epoch {epoch}"): 
                    input = input.to(device=device, dtype=dtype)
                    fake = G(input) 
                    
                    img_name = name[0]
                    save_path = os.path.join(opt.track_train_dir, img_name, f'{epoch:04d}_fake.png')
                    manager.save_image(fake, path=save_path)

                # 2. RUN Track_Test
                for input, _, _, name in tqdm(track_test_loader, desc=f"Tracking Test @ Epoch {epoch}"): 
                    input = input.to(device=device, dtype=dtype)
                    fake = G(input) 
                    
                    img_name = name[0] # Get filename string
                    # Save to the image-specific folder, named by epoch
                    save_path = os.path.join(opt.track_test_dir, img_name, f'{epoch:04d}_fake.png')
                    manager.save_image(fake, path=save_path)
            
            G.train()
            
            print("--- Epoch Tracking Complete ---")
      


        G_scheduler.step() 
        D_scheduler.step()
    print("Total time taken: ", datetime.datetime.now() - start_time)

    # Save final model
    final_save_path = os.path.join(opt.model_dir, 'final_model_dict.pt')
    final_package = {
        'Epoch': opt.n_epochs,
        'current_step': current_step,
        'total_step': total_step,
        'D_state_dict': D.state_dict(),
        'G_state_dict': G.state_dict(),
        'D_optim_state_dict': D_optim.state_dict(),
        'G_optim_state_dict': G_optim.state_dict(),
        'G_scheduler_state_dict': G_scheduler.state_dict(),
        'D_scheduler_state_dict': D_scheduler.state_dict()
    }
    
    print(f"\nSaving final model to: {final_save_path}")
    torch.save(final_package, final_save_path)
    print("Training completed and final model saved successfully!")
