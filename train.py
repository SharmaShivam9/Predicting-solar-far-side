if __name__ == '__main__':
    import torch
    import torch.nn as nn
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
    from torch.utils.data.distributed import DistributedSampler
    import torch.distributed as dist
    import sys
    
    torch.backends.cudnn.benchmark = True

    opt = TrainOption().parse()

# --- STRICT DDP INITIALIZATION (NO FALLBACKS) ---
    if "LOCAL_RANK" not in os.environ:
        print("ERROR: DDP unavailable — LOCAL_RANK not found in environment.")
        sys.exit(1)

    if "WORLD_SIZE" not in os.environ:
        print("ERROR: DDP unavailable — WORLD_SIZE not found in environment.")
        sys.exit(1)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    try:
        dist.init_process_group(backend="nccl", init_method="env://")
    except Exception as e:
        print(f"ERROR: Failed to initialize DDP: {e}")
        sys.exit(1)

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.float32

    is_master = (rank == 0)
    
    def get_state_dict(model):
        return model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    if opt.val_during_train:
        from options import TestOption
        test_opt = TestOption().parse()
        save_freq = opt.save_freq
        
        test_dataset = CustomDataset(test_opt, mode='Test')
        

        if is_master:
            test_data_loader = DataLoader(
                dataset=test_dataset,
                batch_size=1,
                num_workers=opt.n_workers,
                shuffle=False,
                pin_memory=True,
                persistent_workers=True)
        else:
            test_data_loader = None

        
        # --- 2. SETUP Track_Train ---
        print("Setting up Track_Train and Track_Test dataloader...")
        track_train_dataset = CustomDataset(test_opt, mode='Track_Train')
        track_test_dataset = CustomDataset(test_opt, mode='Track_Test')

        
        if is_master:
            track_train_loader = DataLoader(
                dataset=track_train_dataset,
                batch_size=1,
                num_workers=opt.n_workers,
                shuffle=False,
                pin_memory=True,
                persistent_workers=True
            )
            track_test_loader = DataLoader(
                dataset=track_test_dataset,
                batch_size=1,
                num_workers=opt.n_workers,
                shuffle=False,
                pin_memory=True,
                persistent_workers=True
            )
        else:
            track_train_loader = None
            track_test_loader = None



        print("Pre-saving REAL images for tracking folders...")
        if rank == 0:
            temp_manager = Manager(test_opt)
        else:
            temp_manager = None 
        
        # Function to un-normalize and save
        def presave_real_images(loader, base_dir):
            iterator = tqdm(loader, desc=f"Pre-saving REALs in {os.path.basename(base_dir)}") if is_master else loader
            for _, target, _, name in iterator:
                img_name = name[0]  # Get filename string
                track_image_dir = os.path.join(base_dir, img_name)
                os.makedirs(track_image_dir, exist_ok=True)

                real_image_path = os.path.join(track_image_dir, f"_REAL_{img_name}.png")
                if not os.path.exists(real_image_path):
                    if temp_manager:
                        temp_manager.save_image(target, path=real_image_path)

        
        with torch.no_grad():
            if temp_manager:
                presave_real_images(track_train_loader, opt.track_train_dir)
                presave_real_images(track_test_loader, opt.track_test_dir)
        
        print("Tracking setup complete.")
        
    else:
        save_freq = opt.save_freq


    init_lr = opt.lr
    lr = opt.lr

    dataset = CustomDataset(opt)

# --- STRICT DDP SAMPLER ---
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=not opt.no_shuffle)

    data_loader = DataLoader(dataset=dataset,
                         batch_size=opt.batch_size,
                         num_workers=opt.n_workers,
                         sampler=train_sampler,
                         pin_memory=True,
                         persistent_workers=True)


# Move unwrapped models to their specific GPU
    G = Generator(opt).apply(weights_init).to(device, dtype=dtype)
    D = Discriminator(opt).apply(weights_init).to(device, dtype=dtype)

# --- STRICT DDP WRAPPING (NO DATAPARALLEL) ---
    G = torch.nn.parallel.DistributedDataParallel(
        G, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    D = torch.nn.parallel.DistributedDataParallel(
        D, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if is_master:
        print(f"DDP active: rank={rank}, local_rank={local_rank}, world={world_size}")


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
        
        if hasattr(G, "module"):
            G.module.load_state_dict(pt_file['G_state_dict'])
        else:
            G.load_state_dict(pt_file['G_state_dict'])

        if hasattr(D, "module"):
            D.module.load_state_dict(pt_file['D_state_dict'])
        else:
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

    manager = Manager(opt, current_step) if is_master else None
    # --------------------------------------------------------------------------


    total_step = opt.n_epochs * len(data_loader)
    start_time = datetime.datetime.now()
    for epoch in range(init_epoch, opt.n_epochs + 1):
        train_sampler.set_epoch(epoch)   
        train_iterator = tqdm(data_loader, desc=f"Train Epoch {epoch}", disable=(not is_master))
        for batch_index,(input, target, _, _) in enumerate(train_iterator):
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
                           'G_state_dict': get_state_dict(G),
                           'D_state_dict': get_state_dict(D),
                           'D_optim_state_dict': D_optim.state_dict(),
                           'G_optim_state_dict': G_optim.state_dict(),
                           'G_scheduler_state_dict': G_scheduler.state_dict(),
                           'D_scheduler_state_dict': D_scheduler.state_dict(),
                           'target_tensor': target_tensor,
                           'generated_tensor': generated_tensor.detach()}
                
            package.update(custom_losses)
        if is_master:
            try:
                train_iterator.close()
            except Exception:
                pass


            # --- FULL_TEST (Runs every 'save_freq' steps) ---
        if world_size > 1: dist.barrier()
        if manager: manager(package)
        FULL_TEST_INTERVAL =30
        if opt.val_during_train and is_master and (epoch % FULL_TEST_INTERVAL == 0):
                    manager.save(package, model=True)
                    G.eval()                    
                    # The base directory (opt.full_test_dir) is set in options.py
                    epoch_test_dir = os.path.join(opt.full_test_dir, f"epoch_{epoch}")
                    os.makedirs(epoch_test_dir, exist_ok=True)

                    with torch.no_grad():
                        test_iterator = tqdm(test_data_loader, desc=f"Full Test @ Epoch {epoch}") if is_master else test_data_loader
                        for input, target, _, name in test_iterator: 
                            input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)
                            fake = G(input)


                            if len(fake.shape) == 2: fake = fake.unsqueeze(0)
                            if len(target.shape) == 2: target = target.unsqueeze(0)
                            # --- End Un-normalization ---
                            if manager:
                                manager.save_image(fake, os.path.join(epoch_test_dir, f"{name[0]}_fake.png"))
                                manager.save_image(target, os.path.join(epoch_test_dir, f"{name[0]}_real.png"))
                        
                    G.train() 

        if opt.val_during_train and is_master:
            G.eval()

            with torch.no_grad():
                    for input, _, _, name in tqdm(track_train_loader, desc=f"Tracking Train @ Epoch {epoch}"): 
                        input = input.to(device=device, dtype=dtype)
                        fake = G(input) 
                        
                        img_name = name[0]
                        save_path = os.path.join(opt.track_train_dir, img_name, f'{epoch:04d}_fake.png')
                        if manager:
                            manager.save_image(fake, path=save_path)

                    # 2. RUN Track_Test
                    for input, _, _, name in tqdm(track_test_loader, desc=f"Tracking Test @ Epoch {epoch}"): 
                        input = input.to(device=device, dtype=dtype)
                        fake = G(input) 
                        
                        img_name = name[0] # Get filename string
                        # Save to the image-specific folder, named by epoch
                        save_path = os.path.join(opt.track_test_dir, img_name, f'{epoch:04d}_fake.png')
                        if manager:
                            manager.save_image(fake, path=save_path)
            
            G.train()      


        G_scheduler.step() 
        D_scheduler.step()
    print("Total time taken: ", datetime.datetime.now() - start_time)

    # Save final model
    final_save_path = os.path.join(opt.model_dir, 'final_model_dict.pt')
    final_package = {
        'Epoch': opt.n_epochs,
        'current_step': current_step,
        'total_step': total_step,
        'D_state_dict': get_state_dict(D),
        'G_state_dict': get_state_dict(G),
        'D_optim_state_dict': D_optim.state_dict(),
        'G_optim_state_dict': G_optim.state_dict(),
        'G_scheduler_state_dict': G_scheduler.state_dict(),
        'D_scheduler_state_dict': D_scheduler.state_dict()
    }
    if is_master:
        print(f"\nSaving final model to: {final_save_path}")
        torch.save(final_package, final_save_path)
        print("Training completed and final model saved successfully!")
