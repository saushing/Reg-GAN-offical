#!/usr/bin/python3

import argparse
import time
from datetime import datetime
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR,ReplayBuffer
from .utils import weights_init_normal,get_config
from .datasets import ImageDataset, ValDataset, PackedNPYSliceDataset, PackedNPYValSliceDataset
from Model.CycleGan import *
from .utils import Resize,ToTensor,smooothing_loss
from torch.utils.tensorboard import SummaryWriter
from .reg import Reg
from torchvision.transforms import RandomAffine,ToPILImage
from .transformer import Transformer_2D
from skimage import measure
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_fn

def tb_image(x):
     """
    Prepare tensor for TensorBoard:
    - (B,C,H,W) -> (C,H,W)
    - [-1,1] -> [0,1]
    - 1 channel -> 3 channels
    """
     if x.dim() == 4:
         x = x[0]
     x = x.detach().float().cpu()
     x = (x.clamp(-1, 1) +1) / 2.0
     if x.shape[0] == 1:
        x = x.repeat(3, 1, 1) 
     return x

class Cyc_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        use_cuda = bool(config.get("cuda",True)) and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print("Using device:", self.device)
        ## def networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).to(self.device)
        self.netD_B = Discriminator(config['input_nc']).to(self.device)
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        if config['regist']:
            self.R_A = Reg(config['size'], config['size'],config['input_nc'],config['input_nc']).to(self.device)
            self.spatial_transform = Transformer_2D().to(self.device)
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_B2A = Generator(config['input_nc'], config['output_nc']).to(self.device)
            self.netD_A = Discriminator(config['input_nc']).to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
            

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        self.input_A = torch.empty(config['batchSize'], config['input_nc'], config['size'], config['size'], device = self.device)
        self.input_B = torch.empty(config['batchSize'], config['output_nc'], config['size'], config['size'], device = self.device)
        self.target_real = torch.ones(1,1, device = self.device)
        self.target_fake = torch.zeros(1,1, device = self.device)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        #Dataset loader
        level = config['noise_level']  # set noise level
        
        base_tf = [ToTensor()
                   ]
        
        if config.get('use_aug', True) and level >0:
            transforms_1 = [ToPILImage(),
                    RandomAffine(degrees=level,translate=[0.02*level, 0.02*level],scale=[1-0.02*level, 1+0.02*level],fill=-1),
                    ToTensor()
                    ]
        
            transforms_2 = [ToPILImage(),
                    RandomAffine(degrees=1,translate=[0.02, 0.02],scale=[0.98, 1.02],fill=-1),
                    ToTensor()
                    ]
        else:
            transforms_1 = base_tf
            transforms_2 = base_tf
        
        val_transforms = [ToTensor()
                          ]
        
        dataset_type = str(config.get("dataset_type", "legacy")).lower()
        x_channel_index = int(config.get("x_channel_index", 0))  # baseline: 0

        if dataset_type == "packed":
            self.dataloader = DataLoader(
                PackedNPYSliceDataset(
                    config["dataroot"], level,
                    transforms_1=transforms_1, transforms_2=transforms_2,
                    unaligned=False,
                    x_channel_index=x_channel_index,
                    crop_size = config["size"],
                    training= True
                ),
                batch_size=config["batchSize"], shuffle=True, num_workers=config["n_cpu"]
            )

            self.val_data = DataLoader(
                PackedNPYValSliceDataset(
                    config["val_dataroot"],
                    transforms_=val_transforms,
                    unaligned=False,
                    x_channel_index=x_channel_index,
                    crop_size= config["size"]
                ),
                batch_size=config["batchSize"], shuffle=False, num_workers=config["n_cpu"]
            )
        else:
            # legacy flat A/B folders
            self.dataloader = DataLoader(
                ImageDataset(
                    config["dataroot"], level,
                    transforms_1=transforms_1, transforms_2=transforms_2,
                    unaligned=False
                ),
                batch_size=config["batchSize"], shuffle=True, num_workers=config["n_cpu"]
            )

            self.val_data = DataLoader(
                ValDataset(
                    config["val_dataroot"],
                    transforms_=val_transforms,
                    unaligned=False
                ),
                batch_size=config["batchSize"], shuffle=False, num_workers=config["n_cpu"]
            )


 
       # Loss plot
        base_log_dir = self.config.get("log_dir", "./runs/CycleGan")
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(base_log_dir,run_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
       
        
    def train(self):
        assert(not self.config.get("bidirect", False)) and self.config.get("regist", False), \
        "This trainer is modified for NC+R only (bidirect must be False, regist must be True)."
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            sum_loss_D_B = 0
            sum_SR_loss = 0
            count = 0
            epoch_start_time = time.time()
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                if self.config['bidirect']:   # C dir
                    if self.config['regist']:    #C + R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        
                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        
                        Trans = self.R_A(fake_B,real_B) 
                        SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        
                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + SR_loss +SM_loss
                        loss_Total.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()
                        
                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ################################### 
                    
                    else: #only  dir:  C
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda']*self.MSE_loss(pred_fake, self.target_real)

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                        loss_Total.backward()
                        self.optimizer_G.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################
                        
                        
                        
                else:                  # s dir :NC
                    if self.config['regist']:    # NC+R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss
                        fake_B = self.netG_A2B(real_A)

                        # Add Image Logging (after fake_B is computed, before backprop)
                        img_interval = int(self.config.get("image_interval", 10))
                        if (epoch % img_interval == 0) and (i == 0):
                            self.writer.add_image('train_sample/real_A', tb_image(real_A), epoch)
                            self.writer.add_image('train_sample/real_B', tb_image(real_B), epoch)
                            self.writer.add_image('train_sample/fake_B', tb_image(fake_B), epoch)

                        Trans = self.R_A(fake_B,real_B) 
                        SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                        pred_fake0 = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_real)
                        ####smooth loss
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        toal_loss = SM_loss+adv_loss+SR_loss
                        self.writer.add_scalar('loss/adv_loss', adv_loss.item(), epoch)
                        self.writer.add_scalar('loss/SM_loss', SM_loss.item(), epoch)
                        self.writer.add_scalar('loss/SR_loss_step', SR_loss.item(), epoch)
                        self.writer.add_scalar('loss/total_loss_step', toal_loss.item(), epoch)
                        toal_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()

                        # ------D_B update------
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_B = self.netG_A2B(real_A)                        
                        pred_fake0 = self.netD_B(fake_B)
                        pred_real = self.netD_B(real_B)
                        loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_fake)+self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        sum_loss_D_B += float(loss_D_B.item())
                        sum_SR_loss += float(SR_loss.item())
                        count += 1
                        if self.global_step % 1000 == 0:
                            self.writer.add_image("img/CBCT", tb_image(real_A), self.global_step)
                            self.writer.add_image("img/fake_CT", tb_image(fake_B), self.global_step)
                            self.writer.add_image("img/real_CT", tb_image(real_B), self.global_step)
                            self.writer.add_image("img/abs_diff", tb_image(torch.abs(fake_B - real_B)), self.global_step)
                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        self.global_step += 1
                        
                        
                        
                    else:        # only NC
                        self.optimizer_G.zero_grad()
                        fake_B = self.netG_A2B(real_A)
                        #### GAN aligin loss
                        pred_fake = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        adv_loss.backward()
                        self.optimizer_G.step()
                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()
                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)
                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        ###################################
                elapsed =  time.time() - epoch_start_time
                batches_done = i + 1
                batches_total = len(self.dataloader)
                eta = (elapsed / batches_done) * (batches_total - batches_done)
                print(f"\rEpoch {epoch+1:03d}/{self.config['n_epochs']:03d}"
                      f"[{batches_done:04d}/{batches_total:04d}]"
                      f"loss_D_B: {sum_loss_D_B/count:.4f} | "
                      f"SR_loss: {sum_SR_loss/count:.4f} | "   
                      f"ETA: {int(eta//60):02d}:{int(eta%60):02d}",
                      end = ""
                )
                print()
                

            mean_loss_D_B = sum_loss_D_B/max(1, count)
            mean_SR_loss = sum_SR_loss/max(1, count)

            self.writer.add_scalar('loss/loss_D_B', mean_loss_D_B, epoch)
            self.writer.add_scalar('loss/SR_loss', mean_SR_loss, epoch)

    #         # Save models checkpoints
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B_V2.pth')
            #torch.save(self.R_A.state_dict(), self.config['save_root'] + 'Regist.pth')
            #torch.save(netD_A.state_dict(), 'output/netD_A_3D.pth')
            #torch.save(netD_B.state_dict(), 'output/netD_B_3D.pth')
            
            
            #############val###############
            self.netG_A2B.eval()
            with torch.no_grad():
                skipped = 0
                MAE = 0
                num = 0
                for j, batch in enumerate(self.val_data):
                    val_A = self.input_A.copy_(batch['A'].to(self.device))
                    val_B_t = self.input_B.copy_(batch['B'].to(self.device))
                    
                    val_fake_t = self.netG_A2B(val_A)
                    val_B = val_B_t.detach().cpu().numpy().squeeze()
                    val_fake = val_fake_t.detach().cpu().numpy().squeeze()

                    MAE += self.MAE(val_fake,val_B)
                    num += 1

                    val_image_interval = int(self.config.get("val_image_interval",10))
                    if (epoch % val_image_interval == 0) and (j == 0):
                        self.writer.add_image('val_sample/real_A', tb_image(val_A), epoch) 
                        self.writer.add_image('val_sample/real_B', tb_image(val_B_t), epoch)
                        self.writer.add_image('val_sample/fake_B', tb_image(val_fake_t), epoch)
                    
                val_mae = MAE / max(1, num)
                print("\nVal MAE:", val_mae)
                self.writer.add_scalar('val/MAE', val_mae, epoch)
                self.writer.add_scalar('val/skipped_empty_silces', skipped, epoch)
                self.netG_A2B.train()

        self.writer.close()        
                    
                         
    def test(self):
    # ---- Load generator checkpoint ----
        ckpt_path = os.path.join(self.config['save_root'], 'netG_A2B_V2.pth')
        state = torch.load(ckpt_path, map_location=self.device)
        self.netG_A2B.load_state_dict(state)
        self.netG_A2B.eval()

    # ---- New TB run for test ----
        from torch.utils.tensorboard import SummaryWriter
        run_name = datetime.now().strftime("TEST_%Y%m%d-%H%M%S")
        test_log_dir = os.path.join(self.config.get("log_dir", "./runs/CycleGan"), run_name)
        test_writer = SummaryWriter(log_dir=test_log_dir)

    # ---- HU de-normalization settings (match preprocessing) ----
        HU_MIN = float(self.config.get("hu_min", -1000))
        HU_MAX = float(self.config.get("hu_max", 2500))
        HU_RANGE = HU_MAX - HU_MIN

        def denorm_to_hu(x_norm: np.ndarray) -> np.ndarray:
            # x_norm in [-1,1]
            return ((x_norm + 1.0) * 0.5) * HU_RANGE + HU_MIN

        def masked_mae_hu(fake_hu: np.ndarray, real_hu: np.ndarray, mask: np.ndarray) -> float:
            diff = np.abs(fake_hu[mask] - real_hu[mask])
            return float(diff.mean()) if diff.size > 0 else float("nan")

        def masked_psnr_hu(fake_hu: np.ndarray, real_hu: np.ndarray, mask: np.ndarray) -> float:
            if mask.sum() == 0:
                return float("nan")
            mse = np.mean((fake_hu[mask] - real_hu[mask]) ** 2)
            if mse < 1.0e-12:
                return 100.0
            # PSNR uses peak-to-peak dynamic range
            return float(20.0 * np.log10(HU_RANGE / np.sqrt(mse)))

        with torch.no_grad():
            MAE_sum = 0.0
            PSNR_sum = 0.0
            SSIM_sum = 0.0
            num = 0

            vis_batches = int(self.config.get("test_vis_batches", 3))
            step = 0

            for i, batch in enumerate(self.val_data):
                # tensors: (B,1,H,W)
                real_A_t = self.input_A.copy_(batch['A'].to(self.device))
                real_B_t = self.input_B.copy_(batch['B'].to(self.device))

                fake_B_t = self.netG_A2B(real_A_t)

                # log a few example images in normalized space (good for display)
                if i < vis_batches:
                    test_writer.add_image("test/real_A", tb_image(real_A_t), step)
                    test_writer.add_image("test/real_B_norm", tb_image(real_B_t), step)
                    test_writer.add_image("test/fake_B_norm", tb_image(fake_B_t), step)
                    test_writer.add_image("test/abs_diff_norm", tb_image(torch.abs(fake_B_t - real_B_t)), step)
                    step += 1

                # ---- metrics in HU ----
                real_B = real_B_t.detach().cpu().numpy().squeeze()   # (H,W) if B=1
                fake_B = fake_B_t.detach().cpu().numpy().squeeze()

                # background convention: real_B == -1 means HU_MIN (air/outside)
                mask = (real_B != -1)

                real_hu = denorm_to_hu(real_B)
                fake_hu = denorm_to_hu(fake_B)

                # SSIM in HU: set background equal to real to avoid edge artifacts, same idea as your old code :contentReference[oaicite:3]{index=3}
                tmp_fake_hu = fake_hu.copy()
                tmp_fake_hu[~mask] = real_hu[~mask]

                mae_hu = masked_mae_hu(fake_hu, real_hu, mask)
                psnr_hu = masked_psnr_hu(fake_hu, real_hu, mask)
                ssim_hu = float(ssim_fn(tmp_fake_hu, real_hu, data_range=HU_RANGE))

                MAE_sum += mae_hu
                PSNR_sum += psnr_hu
                SSIM_sum += ssim_hu
                num += 1

            MAE_m = MAE_sum / max(1, num)
            PSNR_m = PSNR_sum / max(1, num)
            SSIM_m = SSIM_sum / max(1, num)

            print("TEST (HU) MAE:", MAE_m)
            print("TEST (HU) PSNR:", PSNR_m)
            print("TEST (HU) SSIM:", SSIM_m)

            test_writer.add_scalar("test_hu/MAE", MAE_m, 0)
            test_writer.add_scalar("test_hu/PSNR", PSNR_m, 0)
            test_writer.add_scalar("test_hu/SSIM", SSIM_m, 0)

        test_writer.flush()
        test_writer.close()

                
    
    def PSNR(self,fake,real):
       x,y = np.where(real!= -1)# Exclude background
       mse = np.mean(((fake[x,y]+1)/2. - (real[x,y]+1)/2.) ** 2 )
       if mse < 1.0e-10:
          return 100
       else:
           PIXEL_MAX = 1
           return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
            
            
    def MAE(self,fake,real):
        x,y = np.where(real!= -1)  # Exclude background
        mae = np.abs(fake[x,y]-real[x,y]).mean()
        return mae/2     #from (-1,1) normaliz  to (0,1)
            

    def save_deformation(self,defms,root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max,x_min = dir_x.max(),dir_x.min()
        y_max,y_min = dir_y.max(),dir_y.min()
        dir_x = ((dir_x-x_min)/(x_max-x_min))*255
        dir_y = ((dir_y-y_min)/(y_max-y_min))*255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_x[tans_x<=150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_y[tans_y<=150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5,tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy) 
