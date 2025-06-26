import os
from torch.utils.data import Dataset, DataLoader
from src import dataset
from src.dataset import Dataloading
from config import configp
from config.configp import get_args  # Corrected import statement
from src.transformations import get_train_transforms, get_val_transforms
import monai
import torch
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor

#from monai.transforms import MapTransform, TransformBackends
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
)
from monai.losses import FocalLoss, DiceLoss, DiceCELoss, DiceFocalLoss
import torch.nn as nn
import logging
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from src import train

# from src.VAE import VAE  # Import the VAE class from src/vae.py
from thop import profile
import os
import shutil
import tempfile
import time
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import set_determinism
from torch.nn import L1Loss
from tqdm import tqdm
from sklearn.manifold import TSNE
from monai.inferers import sliding_window_inference

from functools import partial
from generative.networks.layers.vector_quantizer import EMAQuantizer, VectorQuantizer


####LDM

import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
#from generative.networks.schedulers import DDPMScheduler
from monai.utils import first, set_determinism
from monai.losses import FocalLoss, DiceLoss, DiceCELoss, DiceFocalLoss
import torch.nn.functional as F
import visdom
from src.train import train_vae, validate_vae, test_vae  # Import the train and validate functions
from monai.losses import DiceLoss

# Initialize Dice loss
# Initialize Visdom
# viz = visdom.Visdom()
import wandb
import cuml
from cuml.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib  # Import matplotlib first
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.cond_train import train_cond, validate_cond, test_cond  # Import the train and validate functions
import yaml
from src.unet_stack import UNet3DResidual_stack
from src.TC import VQVAE_seq_TC
from src.WT import VQVAE_seq_WT
from src.ET import VQVAE_seq_ET
from monai.transforms import NormalizeIntensityd
from skimage.transform import resize
# from torch.profiler import profile, record_function, ProfilerActivity
wandb.init(
    project="BTS_VAE_Model"
    
)





def run_pipeline(args):
    # Create datasets with splits
   # Create dataset instances for each split
    dice_loss = DiceLoss(to_onehot_y=False, softmax=False)
    #l1_loss = DiceLoss
    data_path=args.data_path
    crop_size=args.crop_size
    modalities=args.modalities
    with open("diff.yaml", "r") as file:
        configd = yaml.safe_load(file)
        print("yaml file loaded")
    print("args_data_path", data_path)


    
    train_dataset, val_dataset, test_dataset=Dataloading("all_data_split.json", crop_size)
    

    # Print out dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)

  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
    print("device is", device)




    
    if args.VQVAE:

        
        model = VQVAE_seq_WT(in_channels=1, out_channels=2, dropout_prob=0.0)
        
        
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-4)

        # Initialize the learning rate scheduler with adjusted parameters
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=200,  # Total epochs before LR reaches min
        eta_min=1e-5  # Minimum LR
    )

    
        
        #model = nn.DataParallel(model.to(device))
        
        model=model.to(device)
        print(model)
        
        if torch.cuda.is_available():
            input = torch.randn(1, 1, 240, 240, 155).to(device)
            flops, params = profile(model, (input,))
            print('Params = ' + str(params/1000**2) + 'M')
            print('FLOPs = ' + str(flops/1000**3) + 'G')


        
        # Create directories for saving model checkpoints
        checkpoint_dir = args.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)  # Create the directory if it does not existmodel,
        
        def visualize_slices(mri_tensor, mask_tensor, pred_mask_tensor, class_losses_sum_overall_wo, hd_95, title='MRI with Segmentation'):
            """
            Visualize the MRI slices with GT and predicted segmentation overlays, including Dice scores and HD95 values.
            
            Args:
                mri_tensor (torch.Tensor): 3D MRI scan with shape [D, H, W].
                mask_tensor (torch.Tensor): 3D GT segmentation mask with shape [D, H, W].
                pred_mask_tensor (torch.Tensor): 3D predicted segmentation mask with shape [D, H, W].
                class_losses_sum_overall_wo (dict): Dictionary containing Dice scores with keys 'WT', 'TC', 'ET'.
                hd_95 (dict): Dictionary containing Hausdorff 95 values with keys 'WT', 'TC', 'ET'.
                title (str): The base title of the plot.
            """
        
            # Convert tensors to numpy
            mask_tensor = mask_tensor.squeeze(dim=0).permute(2, 0, 1)
            mask = mask_tensor.cpu().numpy()
            mri_tensor = mri_tensor.squeeze(dim=0).permute(2, 0, 1)
            mri = mri_tensor.cpu().numpy()
            pred_mask_tensor = pred_mask_tensor.squeeze(dim=0).permute(2, 0, 1)
            pred_mask = pred_mask_tensor.cpu().numpy()
        
            # Define a colormap for segmentation (ED: Green, NC: Red, ET: Yellow)
            cmap = ListedColormap(['black', 'green', 'red', 'yellow'])
        
            # Select two slices for visualization (slice_idx and slice_idx+1)
            num_slices = mri.shape[0]
            slice_idx = num_slices // 3
            slice_idx_2 = num_slices // 2  # Next slice
        
            # Create a 2-row, 6-column subplot
            fig, axs = plt.subplots(2, 6, figsize=(18, 6))
            plt.subplots_adjust(wspace=0.025, hspace=0.025)  # Reduce space between subplots
        
            # Get MRI and mask slices for both slice_idx and slice_idx_2
            slices = [slice_idx, slice_idx_2]
            for i, idx in enumerate(slices):
                mri_slice = mri[idx]
                mask_slice = mask[idx]
                pred_mask_slice = pred_mask[idx]
        
                # Normalize MRI to [0,1]
                mri_slice_norm = (mri_slice - np.min(mri_slice)) / (np.max(mri_slice) - np.min(mri_slice))
        
                # Create an RGB version of the grayscale MRI
                mri_rgb = np.stack([mri_slice_norm] * 3, axis=-1)
        
                # Apply tumor colors to the GT and predicted masks
                def apply_colormap(slice_image, mask):
                    overlay = np.copy(slice_image)
                    overlay[mask == 1] = cmap(1)[:3]  # Green
                    overlay[mask == 2] = cmap(2)[:3]  # Red
                    overlay[mask == 3] = cmap(3)[:3]  # Yellow
                    return overlay

                def crop_and_resize(image, crop_factor=0.52):
                    """
                    Crop the center region and resize it back to original dimensions.
                    Args:
                        image (numpy array): The input image.
                        crop_factor (float): The fraction of the image size to crop (default: 0.5 means 50% smaller).
                    Returns:
                        Resized zoomed-in image.
                    """
                    H, W, _ = image.shape  # Get height and width
                    ch, cw = int(H * crop_factor), int(W * crop_factor)  # Compute crop size
                    
                    # Crop center
                    start_x = (H - ch) // 2
                    start_y = (W - cw) // 2
                    cropped = image[start_x:start_x+ch, start_y:start_y+cw]
            
                    # Resize back to original size
                    zoomed_in = resize(cropped, (H, W), anti_aliasing=True)
            
                    return zoomed_in
        
                # Generate overlays
                gt_mask_overlay = apply_colormap(mri_rgb, mask_slice)
                pred_mask_overlay = apply_colormap(mri_rgb, pred_mask_slice)
        
                # Display the original images in the top row
                axs[0, i * 3].imshow(mri_rgb)
                axs[0, i * 3].axis('off')
        
                axs[0, i * 3 + 1].imshow(gt_mask_overlay)
                axs[0, i * 3 + 1].axis('off')
        
                axs[0, i * 3 + 2].imshow(pred_mask_overlay)
                axs[0, i * 3 + 2].axis('off')
        
                # Apply zoom-in function to images
                zoomed_mri = crop_and_resize(mri_rgb)
                zoomed_gt = crop_and_resize(gt_mask_overlay)
                zoomed_pred = crop_and_resize(pred_mask_overlay)
        
                # Display the zoomed-in images in the bottom row
                axs[1, i * 3].imshow(zoomed_mri)
                axs[1, i * 3].axis('off')
        
                axs[1, i * 3 + 1].imshow(zoomed_gt)
                axs[1, i * 3 + 1].axis('off')
        
                axs[1, i * 3 + 2].imshow(zoomed_pred)
                axs[1, i * 3 + 2].axis('off')
        
            # Set the main title with Dice scores
            dice_wt = class_losses_sum_overall_wo.get('WT', 0.0)
            dice_tc = class_losses_sum_overall_wo.get('TC', 0.0)
            dice_et = class_losses_sum_overall_wo.get('ET', 0.0)
            # avg_dice = (dice_wt + dice_tc + dice_et) / 3
            hd_wt = hd_95.get('WT_HD_95', 0.0)
            hd_tc = hd_95.get('TC_HD_95', 0.0)
            hd_et = hd_95.get('ET_HD_95', 0.0)
            # Save the plot
            plot_path = f"{title.replace(' ', '_')}_overlay.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
        
            # Log to W&B
            wandb.log({f"{title}_overlay": wandb.Image(plot_path)})
        
        # Determine whether to resume training or start from scratch
        if args.vqvae_training:
            
            start_epoch = 0  # Default start epoch
            if args.resume:
                print("Resume training from epoch")
                checkpoint_path = os.path.join(checkpoint_dir, 'vae_checkpoint_latest_WT.pth')
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path)  # Load the latest checkpoint
                    print("checkpoint.keys()", checkpoint['model_state_dict'].keys())
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Restore model state
                    optimizer.load_state_dict(checkpoint['optimizer1_state_dict'])  # Restore optimizer state
                    start_epoch = checkpoint['epoch']  # Restore the last epoch
                    print(f"Resumed from epoch {start_epoch} with train loss {checkpoint['train_loss']} and val loss {checkpoint['val_loss']} and optimizer {checkpoint['optimizer1_state_dict']} ")
            else:
                print("No checkpoint found. Starting training from scratch.")
                checkpoint_path = os.path.join(checkpoint_dir, 'vae_checkpoint_latest_WT.pth')
                # if os.path.exists(checkpoint_path):
                #     os.remove(checkpoint_path)  # Remove existing checkpoint if starting from scratch
        
            # Training and validation loop
            total_start = time.time()
            num_epochs = 2000  # Set the number of epochs

            model_inferer = partial(
            sliding_window_inference,
            roi_size=crop_size,
            sw_batch_size=args.batch_size,
            predictor=model,
            overlap=0.5,
        )
            
            train_loss_lis=[]
            val_loss_lis=[]

            
            def plot_encodings_sum(encodings_sumation, epoch):
                """Plot all 1024 encoding sums as a bar chart on the same figure and log to WandB."""
                # Move the tensor to the CPU, check its shape, and convert to numpy
                encodings_sumation_cpu = encodings_sumation.cpu().numpy()
            
                # Debugging: Print the shape to ensure it has 1024 values
                print(f"Shape of encodings_sumation at epoch {epoch}: {encodings_sumation_cpu.shape}")
            
                # If the tensor is not 1D, flatten it
                if len(encodings_sumation_cpu.shape) > 1:
                    encodings_sumation_cpu = encodings_sumation_cpu.flatten()
                
                # Plot all 1024 values as a bar chart
                plt.figure(figsize=(15, 6))  # Adjust figure size for better visibility
                plt.bar(range(len(encodings_sumation_cpu)), encodings_sumation_cpu)
                plt.xlabel("Encoding Index")
                plt.ylabel("Sum of Encodings")
                plt.title(f"Encodings Sum at Epoch {epoch}")
            
                # Log the figure to WandB as an image
                wandb.log({f'encodings_sum_epoch_{epoch}': wandb.Image(plt)})
                
                # Close the plot to free memory
                plt.close()

            def log_latent_space(latents, latent_name, epoch, n_clusters=4):
                """
                Visualizes and logs each latent space separately to W&B using GPU-based t-SNE and Seaborn scatter plot.
                
                Args:
                latents: The latent vectors extracted from a specific codebook, shape (batch_size, num_channels, height, width, depth)
                latent_name: The name to log for each quantized latent (e.g., z_quantized0)
                epoch: Current epoch to track the training phase
                n_clusters: Number of clusters for coloring the latent points
                """
                # Reshape latents to (batch_size * spatial dimensions, num_channels)
                latents_reshaped = latents.view(-1, latents.size(1))  # Shape (batch_size * 30 * 30 * 20, num_channels)
            
                # Detach from computation graph before converting to NumPy
                latents_reshaped = latents_reshaped.detach().cpu().numpy()
            
                # Check the number of samples in reshaped latents
                num_samples = latents_reshaped.shape[0]
            
                # Set the perplexity, ensure it is smaller than the number of samples
                perplexity = min(30, num_samples - 1)
            
                # Apply GPU-based t-SNE to reduce the high-dimensional latent space to 2D
                tsne = cuml.TSNE(n_components=2, perplexity=perplexity)
            
                try:
                    latents_2d = tsne.fit_transform(latents_reshaped)
            
                    # Apply KMeans to find clusters in the latent space
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                    cluster_labels = kmeans.fit_predict(latents_reshaped)
            
                    # Create a DataFrame for easier plotting
                    df = pd.DataFrame(latents_2d, columns=['x', 'y'])
                    df['cluster'] = cluster_labels
            
                    # Create the scatter plot using Seaborn
                    plt.figure(figsize=(10, 8))
                    sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='viridis', style='cluster', markers='o', s=100)
                    plt.title(f"{latent_name} Latent Space at Epoch {epoch}")
                    plt.xlabel('TSNE Component 1')
                    plt.ylabel('TSNE Component 2')
            
                    # Save the plot to a temporary file
                    plot_path = f"{latent_name}_latent_space_epoch.png"
                    plt.savefig(plot_path)
                    plt.close()
            
                    # Log the image to W&B
                    wandb.log({f"{latent_name}_latent_space": wandb.Image(plot_path)})
            
                except ValueError as e:
                    print(f"Error visualizing latent space for {latent_name}: {e}")
        

            utilized_encoding = torch.zeros(512).to(device)
            for epoch in range(start_epoch, num_epochs):  # Resume from the last saved epoch or start from scratch
                print("Training in progress")
                print(f"Epoch {epoch+1}/{num_epochs}")  # Print current epoch number
                
                train_loss, class_losses_sum_overall, class_losses_sum_overall_wo = train_vae(model, train_loader, len(train_dataset), optimizer, device)
                wandb.log({'epoch': epoch + 1, 'train_loss': train_loss / 2})
                wandb.log({'epoch': epoch + 1, 'class_losses_sum_train_overall': class_losses_sum_overall})
                wandb.log({'epoch': epoch + 1, 'class_losses_sum_train_overall_wo': class_losses_sum_overall_wo})
                # train_loss_lis.append(train_loss)
                print(f'Train Loss: {train_loss:.4f}')  # Print training loss
                # print(f'class_losses_sum_train_overall: {class_losses_sum_overall}')
                torch.cuda.empty_cache()
                
                # Validate the model every 5 epochs
                if (epoch + 1) % 1 == 0:
                    print("Validation in progress")
                    # # mask_validation, reconstruction_validation, mask_val, val_loss, class_losses_sum_overall_val, class_losses_sum_overall_val_wo, Q_loss_val, embedding0 = validate_vae(cond_model, model, model_inferer, val_loader, len(val_dataset), device)
                    val_loss, class_losses_sum_overall_val, class_losses_sum_overall_val_wo = validate_vae(model, model_inferer, test_loader, len(test_dataset), device)
                   
                    wandb.log({'epoch': epoch + 1, 'val_loss': val_loss / 2})
                    wandb.log({'epoch': epoch + 1, 'class_losses_sum_val_overall_val': class_losses_sum_overall_val})
                    wandb.log({'epoch': epoch + 1, 'class_losses_sum_overall_val_wo': class_losses_sum_overall_val_wo})
                    # wandb.log({'epoch': epoch + 1, 'hd_95es': hd_95es})

                    val_loss, class_losses_sum_overall_val, class_losses_sum_overall_val_wo = validate_vae(model, model_inferer, val_loader, len(val_dataset), device)
                   
                    wandb.log({'epoch': epoch + 1, 'val_loss': val_loss / 2})
                    wandb.log({'epoch': epoch + 1, 'class_losses_sum_val_overall_val': class_losses_sum_overall_val})
                    wandb.log({'epoch': epoch + 1, 'class_losses_sum_overall_val_wo': class_losses_sum_overall_val_wo})
                    # wandb.log({'epoch': epoch + 1, 'hd_95es': hd_95es})
                    # wandb.log({'epoch': epoch + 1, 'Q_loss_val': Q_loss_val})
                    # wandb.log({'epoch': epoch + 1, 'total_encodings': total_encodings})
                    # log_latent_space(mask_val, "z_quantized_all", epoch + 1)
                    # wandb.log({'epoch': epoch + 1, 'val_miss_loss': val_miss_loss})
                    print(f'Validation Loss: {val_loss:.4f}')  # Print validation loss
                    # print(f'class_losses_sum_train_overall: {class_losses_sum_overall}')
                    # scheduler.step()
        
                    # # Save checkpoint to resume training if needed
                    torch.save({
                        'epoch': epoch + 1,  # Save current epoch
                        'model_state_dict': model.state_dict(),  # Save model state
                        'optimizer1_state_dict': optimizer.state_dict(),  # Save optimizer state
                        # 'Utilized_encoding' : utilized_encoding,
                        # 'embedding0' : embedding0,
                        # 'optimizer2_state_dict': optimizer2.state_dict(),
                        'train_loss': train_loss,  # Save training loss
                        'val_loss': val_loss,  # Save validation loss
                        'train_loss_list': train_loss_lis,
                        'val_loss_lis': val_loss_lis
                    }, checkpoint_path)
                    print(f'Saved checkpoint to {checkpoint_path}')  # Confirm checkpoint saving
                    torch.cuda.empty_cache()
                if (epoch + 1) % 1000 == 0:
                    
                    mask_test, reconstruction_test, mask_test_qnt, test_loss, class_losses_sum_overall_test, class_losses_sum_overall_test_wo, Q_loss_test, embedding0 = validate_vae(model, model_inferer, test_loader, len(test_dataset), device)
                    print('class_losses_sum_val_overall_test', class_losses_sum_overall_test)
                    wandb.log({'epoch': epoch + 1, 'test_loss': test_loss / 5})
                    wandb.log({'epoch': epoch + 1, 'class_losses_sum_test_overall_test': class_losses_sum_overall_test})
                    wandb.log({'epoch': epoch + 1, 'class_losses_sum_overall_test_wo': class_losses_sum_overall_test_wo})
                    wandb.log({'epoch': epoch + 1, 'Q_loss_test': Q_loss_test})
            total_time = time.time() - total_start
            print(f"train completed, total time: {total_time}.")
            
        else:
            model_inferer = partial(
            sliding_window_inference,
            roi_size=crop_size,
            sw_batch_size=args.batch_size,
            predictor=model,
            overlap=0.5,
        )
            print("No Training argument provided")
           
        #     checkpoint_path = os.path.join(checkpoint_dir, 'vae_checkpoint_latest_WT.pth')
        #     checkpoint = torch.load(checkpoint_path)
            
    
        #     model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        #     print(torch.cuda.is_available())
        #     print("device is", device)
        #     model.to(device) 


            autoencoder_WT = VQVAE_seq_WT(in_channels=1, out_channels=2, dropout_prob=0.0)

            autoencoder_WT.to(device)

            
    
            checkpoint_dir = args.checkpoint_dir
    
            checkpoint_path = os.path.join(checkpoint_dir, 'vae_checkpoint_latest_WT.pth')
            checkpoint = torch.load(checkpoint_path)
    
            # pre_trained_embedding = checkpoint['embedding0']
    
            # print("pre_trained_embedding size is", pre_trained_embedding.shape)
    
    
            autoencoder_WT.load_state_dict(checkpoint['model_state_dict'], strict=False)
                  
            
            autoencoder_WT.eval()
            for param in autoencoder_WT.parameters():
                param.requires_grad = False
    
            print("DONE with VQVAE Model loading")


            autoencoder_TC = VQVAE_seq_TC(in_channels=1, out_channels=2, dropout_prob=0.0)

            autoencoder_TC.to(device)
           
            checkpoint_dir = args.checkpoint_dir
    
            checkpoint_path = os.path.join(checkpoint_dir, 'vae_checkpoint_latest_TC.pth')
            checkpoint = torch.load(checkpoint_path)
    
            
    
    
            autoencoder_TC.load_state_dict(checkpoint['model_state_dict'], strict=False)
                  
            
            autoencoder_TC.eval()
            for param in autoencoder_TC.parameters():
                param.requires_grad = False
    
            print("DONE with VQVAE Model loading")


            autoencoder_ET = VQVAE_seq_ET(in_channels=1, out_channels=2, dropout_prob=0.0)

            autoencoder_ET.to(device)
            
    
            checkpoint_dir = args.checkpoint_dir
    
            checkpoint_path = os.path.join(checkpoint_dir, 'vae_checkpoint_latest_ET.pth')
            checkpoint = torch.load(checkpoint_path)
    
            
    
            autoencoder_ET.load_state_dict(checkpoint['model_state_dict'], strict=False)
                  
            
            autoencoder_ET.eval()
            for param in autoencoder_ET.parameters():
                param.requires_grad = False
    
            print("DONE with VQVAE Model loading")
            
            

            def log_latent_space(latents, latent_name, epoch, n_clusters=2):
                """
                Visualizes and logs each latent space separately to W&B using GPU-based t-SNE and Seaborn scatter plot.
                
                Args:
                latents: The latent vectors extracted from a specific codebook, shape (batch_size, num_channels, height, width, depth)
                latent_name: The name to log for each quantized latent (e.g., z_quantized0)
                epoch: Current epoch to track the training phase
                n_clusters: Number of clusters for coloring the latent points
                """
                # Reshape latents to (batch_size * spatial dimensions, num_channels)
                latents_reshaped = latents.view(-1, latents.size(1))  # Shape (batch_size * 30 * 30 * 20, num_channels)
            
                # Detach from computation graph before converting to NumPy
                latents_reshaped = latents_reshaped.detach().cpu().numpy()
            
                # Check the number of samples in reshaped latents
                num_samples = latents_reshaped.shape[0]
            
                # Set the perplexity, ensure it is smaller than the number of samples
                perplexity = min(30, num_samples - 1)
            
                # Apply GPU-based t-SNE to reduce the high-dimensional latent space to 2D
                tsne = cuml.TSNE(n_components=2, perplexity=perplexity)
            
                try:
                    latents_2d = tsne.fit_transform(latents_reshaped)
            
                    # Apply KMeans to find clusters in the latent space
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                    cluster_labels = kmeans.fit_predict(latents_reshaped)
            
                    # Create a DataFrame for easier plotting
                    df = pd.DataFrame(latents_2d, columns=['x', 'y'])
                    df['cluster'] = cluster_labels
            
                    # Create the scatter plot using Seaborn
                    plt.figure(figsize=(10, 8))
                    sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='viridis', style='cluster', markers='o', s=100)
                    plt.title(f"{latent_name} Latent Space at Epoch {epoch}")
                    plt.xlabel('TSNE Component 1')
                    plt.ylabel('TSNE Component 2')
            
                    # Save the plot to a temporary file
                    plot_path = f"{latent_name}_latent_space_epoch.png"
                    plt.savefig(plot_path)
                    plt.close()
            
                    # Log the image to W&B
                    wandb.log({f"{latent_name}_latent_space": wandb.Image(plot_path)})
            
                except ValueError as e:
                    print(f"Error visualizing latent space for {latent_name}: {e}")

            mask_validation, reconstruction_validation, t2f, class_losses_sum_overall_wo, hd_95 = test_vae(autoencoder_WT, autoencoder_TC, autoencoder_ET, model_inferer, test_loader, len(test_dataset), device)

    elif args.COND:

        
        autoencoder = VQVAE_seq_WT(in_channels=1, out_channels=4, dropout_prob=0.0)

        autoencoder.to(device)
        # print(model)
        
        if torch.cuda.is_available():
            input = torch.randn(1,1, 240, 240, 155).to(device)
            flops, params = profile(autoencoder, (input,))
            print('Params = ' + str(params/1000**2) + 'M')
            print('FLOPs = ' + str(flops/1000**3) + 'G')

        checkpoint_dir = args.checkpoint_dir

        checkpoint_path = os.path.join(checkpoint_dir, 'vae_checkpoint_latest_WT.pth')
        checkpoint = torch.load(checkpoint_path)

        # pre_trained_embedding = checkpoint['embedding0']

        # print("pre_trained_embedding size is", pre_trained_embedding.shape)


        autoencoder.load_state_dict(checkpoint['model_state_dict'])
              
        
        autoencoder.eval()
        for param in autoencoder.parameters():
            param.requires_grad = False

        print("DONE with VQVAE Model loading")
        
        
        print("Cond VQVAE model Loading")
        model = UNet3DResidual_stack(autoencoder.quantizer0, autoencoder.conv3, autoencoder.conv4, autoencoder.decoder, autoencoder.segmentation, in_channels=4, out_channels=1, dropout_prob=0.0)

        
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-4)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-6)


        # Initialize the learning rate scheduler with adjusted parameters
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=200,  # Total epochs before LR reaches min
        eta_min=1e-6  # Minimum LR
    )


    
        
        #model = nn.DataParallel(model.to(device))
        
        model=model.to(device)
        print(model)
        
        if torch.cuda.is_available():
            input = torch.randn(1, 4, 240, 240, 155).to(device)
            flops, params = profile(model, (input, ))
            print('Params = ' + str(params/1000**2) + 'M')
            print('FLOPs = ' + str(flops/1000**3) + 'G')


       
    
        # Create directories for saving model checkpoints
        checkpoint_dir = args.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)  # Create the directory if it does not existmodel,



        def visualize_slices(mri_tensor, mask_tensor, pred_mask_tensor, class_losses_sum_overall_wo, hd_95, title='MRI with Segmentation'):
            """
            Visualize the MRI slices with GT and predicted segmentation overlays, including Dice scores and HD95 values.
            
            Args:
                mri_tensor (torch.Tensor): 3D MRI scan with shape [D, H, W].
                mask_tensor (torch.Tensor): 3D GT segmentation mask with shape [D, H, W].
                pred_mask_tensor (torch.Tensor): 3D predicted segmentation mask with shape [D, H, W].
                class_losses_sum_overall_wo (dict): Dictionary containing Dice scores with keys 'WT', 'TC', 'ET'.
                hd_95 (dict): Dictionary containing Hausdorff 95 values with keys 'WT', 'TC', 'ET'.
                title (str): The base title of the plot.
            """
        
            # Convert tensors to numpy
            mask_tensor = mask_tensor.squeeze(dim=0).permute(2, 0, 1)
            mask = mask_tensor.cpu().numpy()
            mri_tensor = mri_tensor.squeeze(dim=0).permute(2, 0, 1)
            mri = mri_tensor.cpu().numpy()
            pred_mask_tensor = pred_mask_tensor.squeeze(dim=0).permute(2, 0, 1)
            pred_mask = pred_mask_tensor.cpu().numpy()
        
            # Define a colormap for segmentation (ED: Green, NC: Red, ET: Yellow)
            cmap = ListedColormap(['black', 'green', 'red', 'yellow'])
        
            # Select two slices for visualization (slice_idx and slice_idx+1)
            num_slices = mri.shape[0]
            slice_idx = num_slices // 3
            slice_idx_2 = num_slices // 2  # Next slice
            slice_idx_3 = 64  # Next slice
        
            # Create a 2-row, 6-column subplot
            fig, axs = plt.subplots(2, 6, figsize=(18, 6))
            plt.subplots_adjust(wspace=0.025, hspace=0.025)  # Reduce space between subplots
        
            # Get MRI and mask slices for both slice_idx and slice_idx_2
            slices = [slice_idx, slice_idx_2]
            for i, idx in enumerate(slices):
                mri_slice = mri[idx]
                mask_slice = mask[idx]
                pred_mask_slice = pred_mask[idx]
        
                # Normalize MRI to [0,1]
                mri_slice_norm = (mri_slice - np.min(mri_slice)) / (np.max(mri_slice) - np.min(mri_slice))
        
                # Create an RGB version of the grayscale MRI
                mri_rgb = np.stack([mri_slice_norm] * 3, axis=-1)
        
                # Apply tumor colors to the GT and predicted masks
                def apply_colormap(slice_image, mask):
                    overlay = np.copy(slice_image)
                    overlay[mask == 1] = cmap(1)[:3]  # Green
                    overlay[mask == 2] = cmap(2)[:3]  # Red
                    overlay[mask == 3] = cmap(3)[:3]  # Yellow
                    return overlay

                def crop_and_resize(image, crop_factor=0.52):
                    """
                    Crop the center region and resize it back to original dimensions.
                    Args:
                        image (numpy array): The input image.
                        crop_factor (float): The fraction of the image size to crop (default: 0.5 means 50% smaller).
                    Returns:
                        Resized zoomed-in image.
                    """
                    H, W, _ = image.shape  # Get height and width
                    ch, cw = int(H * crop_factor), int(W * crop_factor)  # Compute crop size
                    
                    # Crop center
                    start_x = (H - ch) // 2
                    start_y = (W - cw) // 2
                    cropped = image[start_x:start_x+ch, start_y:start_y+cw]
            
                    # Resize back to original size
                    zoomed_in = resize(cropped, (H, W), anti_aliasing=True)
            
                    return zoomed_in
        
                # Generate overlays
                gt_mask_overlay = apply_colormap(mri_rgb, mask_slice)
                pred_mask_overlay = apply_colormap(mri_rgb, pred_mask_slice)
        
                # Display the original images in the top row
                axs[0, i * 3].imshow(mri_rgb)
                axs[0, i * 3].axis('off')
        
                axs[0, i * 3 + 1].imshow(gt_mask_overlay)
                axs[0, i * 3 + 1].axis('off')
        
                axs[0, i * 3 + 2].imshow(pred_mask_overlay)
                axs[0, i * 3 + 2].axis('off')
        
                # Apply zoom-in function to images
                zoomed_mri = crop_and_resize(mri_rgb)
                zoomed_gt = crop_and_resize(gt_mask_overlay)
                zoomed_pred = crop_and_resize(pred_mask_overlay)
        
                # Display the zoomed-in images in the bottom row
                axs[1, i * 3].imshow(zoomed_mri)
                axs[1, i * 3].axis('off')
        
                axs[1, i * 3 + 1].imshow(zoomed_gt)
                axs[1, i * 3 + 1].axis('off')
        
                axs[1, i * 3 + 2].imshow(zoomed_pred)
                axs[1, i * 3 + 2].axis('off')
        
            # Set the main title with Dice scores
            dice_wt = class_losses_sum_overall_wo.get('WT', 0.0)
            dice_tc = class_losses_sum_overall_wo.get('TC', 0.0)
            dice_et = class_losses_sum_overall_wo.get('ET', 0.0)
            # avg_dice = (dice_wt + dice_tc + dice_et) / 3
            hd_wt = hd_95.get('WT_HD_95', 0.0)
            hd_tc = hd_95.get('TC_HD_95', 0.0)
            hd_et = hd_95.get('ET_HD_95', 0.0)
            # avg_hd = (hd_wt + hd_tc + hd_et) / 3
        
            # Annotate Dice scores and HD95 values
            # axs[1, 0].text(10, 20, f"dice_wt: {dice_wt:.2f}", color='white', fontsize=12, weight='bold')
            # axs[1, 0].text(10, 40, f"dice_tc: {dice_tc:.2f}", color='white', fontsize=12, weight='bold')
            # axs[1, 0].text(10, 60, f"dice_et: {dice_et:.2f}", color='white', fontsize=12, weight='bold')
        
            
        
            # axs[1, 3].text(10, 20, f"HD95 WT: {hd_wt:.2f}", color='white', fontsize=12, weight='bold')
            # axs[1, 3].text(10, 40, f"HD95 TC: {hd_tc:.2f}", color='white', fontsize=12, weight='bold')
            # axs[1, 3].text(10, 60, f"HD95 ET: {hd_et:.2f}", color='white', fontsize=12, weight='bold')
        
            # Save the plot
            plot_path = f"{title.replace(' ', '_')}_overlay.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
        
            # Log to W&B
            wandb.log({f"{title}_overlay": wandb.Image(plot_path)})
        

        # Determine whether to resume training or start from scratch
        if args.cond_training:
            
            start_epoch = 0  # Default start epoch
            if args.resume:
                print("Resume training from epoch")
                checkpoint_path = os.path.join(checkpoint_dir, 'cond_vae_checkpoint_latest_WT.pth')
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path)  # Load the latest checkpoint
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Restore model state
                    # optimizer.load_state_dict(checkpoint['optimizer1_state_dict'])  # Restore optimizer state
                    start_epoch = checkpoint['epoch']  # Restore the last epoch
                    print(f"Resumed from epoch {start_epoch} with train loss {checkpoint['train_loss']} and val loss {checkpoint['val_loss']} and optimizer {checkpoint['optimizer1_state_dict']} ")
            else:
                print("No checkpoint found. Starting training from scratch.")
                checkpoint_path = os.path.join(checkpoint_dir, 'cond_vae_checkpoint_latest_WT.pth')
                # if os.path.exists(checkpoint_path):
                #     os.remove(checkpoint_path)  # Remove existing checkpoint if starting from scratch
        
            # Training and validation loop
            total_start = time.time()
            num_epochs = 700  # Set the number of epochs
            # model_inferer = partial(
            #     sliding_window_inference,
            #     roi_size=crop_size,
            #     sw_batch_size=4,
            #     predictor=model,
            #     overlap=0.5,
            # )
            model_inferer = partial(
            sliding_window_inference,
            roi_size=crop_size,
            sw_batch_size=args.batch_size,
            predictor=model,
            overlap=0.5,
        )
            
            train_loss_lis=[]
            val_loss_lis=[]

            
            def plot_encodings_sum(encodings_sumation, epoch):
                """Plot all 1024 encoding sums as a bar chart on the same figure and log to WandB."""
                # Move the tensor to the CPU, check its shape, and convert to numpy
                encodings_sumation_cpu = encodings_sumation.cpu().numpy()
            
                # Debugging: Print the shape to ensure it has 1024 values
                print(f"Shape of encodings_sumation at epoch {epoch}: {encodings_sumation_cpu.shape}")
            
                # If the tensor is not 1D, flatten it
                if len(encodings_sumation_cpu.shape) > 1:
                    encodings_sumation_cpu = encodings_sumation_cpu.flatten()
                
                # Plot all 1024 values as a bar chart
                plt.figure(figsize=(15, 6))  # Adjust figure size for better visibility
                plt.bar(range(len(encodings_sumation_cpu)), encodings_sumation_cpu)
                plt.xlabel("Encoding Index")
                plt.ylabel("Sum of Encodings")
                plt.title(f"Encodings Sum at Epoch {epoch}")
            
                # Log the figure to WandB as an image
                wandb.log({f'encodings_sum_epoch_{epoch}': wandb.Image(plt)})
                
                # Close the plot to free memory
                plt.close()

            def log_latent_space(latents, latent_name, epoch, n_clusters=512):
                """
                Visualizes and logs each latent space separately to W&B using GPU-based t-SNE and Seaborn scatter plot.
                
                Args:
                latents: The latent vectors extracted from a specific codebook, shape (batch_size, num_channels, height, width, depth)
                latent_name: The name to log for each quantized latent (e.g., z_quantized0)
                epoch: Current epoch to track the training phase
                n_clusters: Number of clusters for coloring the latent points
                """
                # Reshape latents to (batch_size * spatial dimensions, num_channels)
                latents_reshaped = latents.view(-1, latents.size(1))  # Shape (batch_size * 30 * 30 * 20, num_channels)
            
                # Detach from computation graph before converting to NumPy
                latents_reshaped = latents_reshaped.detach().cpu().numpy()
            
                # Check the number of samples in reshaped latents
                num_samples = latents_reshaped.shape[0]
            
                # Set the perplexity, ensure it is smaller than the number of samples
                perplexity = min(30, num_samples - 1)
            
                # Apply GPU-based t-SNE to reduce the high-dimensional latent space to 2D
                tsne = cuml.TSNE(n_components=2, perplexity=perplexity)
            
                try:
                    latents_2d = tsne.fit_transform(latents_reshaped)
            
                    # Apply KMeans to find clusters in the latent space
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                    cluster_labels = kmeans.fit_predict(latents_reshaped)
            
                    # Create a DataFrame for easier plotting
                    df = pd.DataFrame(latents_2d, columns=['x', 'y'])
                    df['cluster'] = cluster_labels
            
                    # Create the scatter plot using Seaborn
                    plt.figure(figsize=(10, 8))
                    sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='viridis', style='cluster', markers='o', s=100)
                    plt.title(f"{latent_name} Latent Space at Epoch {epoch}")
                    plt.xlabel('TSNE Component 1')
                    plt.ylabel('TSNE Component 2')
            
                    # Save the plot to a temporary file
                    plot_path = f"{latent_name}_latent_space_epoch.png"
                    plt.savefig(plot_path)
                    plt.close()
            
                    # Log the image to W&B
                    wandb.log({f"{latent_name}_latent_space": wandb.Image(plot_path)})
            
                except ValueError as e:
                    print(f"Error visualizing latent space for {latent_name}: {e}")
        

            utilized_encoding = torch.zeros(512).to(device)
            best_hd95 = float('inf')  # Initialize with a high value
            for epoch in range(start_epoch, num_epochs):  # Resume from the last saved epoch or start from scratch
                print("Training in progress")
                print(f"Epoch {epoch+1}/{num_epochs}")  # Print current epoch number
                
                latent_loss, class_losses_sum_overall_wo = train_cond(model, autoencoder, train_loader, len(train_dataset), optimizer, device)
                
                wandb.log({'epoch': epoch + 1, 'latent_loss': latent_loss})
                # wandb.log({'epoch': epoch + 1, 'class_losses_sum_val_overall_val': class_losses_sum_overall})
                wandb.log({'epoch': epoch + 1, 'class_losses_sum_overall_wo': class_losses_sum_overall_wo})
                
                torch.cuda.empty_cache()
                # Validate the model every 5 epochs
                if (epoch + 1) % 1 == 0:
                    print("Validation in progress")
                    latent_loss_val, class_losses_sum_overall_wo, class_losses_sum_overall, hd_95es = validate_cond(model, autoencoder, val_loader, len(val_dataset), device)
        
                   
                    wandb.log({'epoch': epoch + 1, 'latent_loss_val': latent_loss_val})
                    wandb.log({'epoch': epoch + 1, 'hd_95es': hd_95es})
                    # wandb.log({'epoch': epoch + 1, 'indices_loss_val': indices_loss_val})
                    wandb.log({'epoch': epoch + 1, 'class_losses_sum_val_overall_val': class_losses_sum_overall})
                    wandb.log({'epoch': epoch + 1, 'class_losses_sum_overall_val_wo': class_losses_sum_overall_wo})
                    
                    scheduler.step()

                    if hd_95es <= best_hd95:  # Save only if hd_95es improves
                        print(f'New best hd_95es found: {hd_95es:.4f} (Previous: {best_hd95:.4f}), saving checkpoint...')
                        best_hd95 = hd_95es  #
            
                        # Save checkpoint to resume training if needed
                        torch.save({
                            'epoch': epoch + 1,  # Save current epoch
                            'model_state_dict': model.state_dict(),  # Save model state
                            'optimizer1_state_dict': optimizer.state_dict(),  # Save optimizer state
                            # 'Utilized_encoding' : utilized_encoding,
                            # 'embedding0' : embedding0,
                            # 'embedding1' : embedding1,
                            # 'embedding2' : embedding2,
                            # 'embedding3' : embedding3,
                            # 'optimizer2_state_dict': optimizer2.state_dict(),
                            'train_loss': latent_loss,  # Save training loss
                            'val_loss': latent_loss_val,  # Save validation loss
                            # 'train_loss_list': train_loss_lis,
                            # 'val_loss_lis': val_loss_lis
                        }, checkpoint_path)
                        print(f'Saved checkpoint to {checkpoint_path}')  # Confirm checkpoint saving
                    torch.cuda.empty_cache()
                if (epoch + 1) % 1000 == 0:
                    
                    for mask_validation, reconstruction_validation, test_loss in test_cond(model, test_loader, len(test_dataset), device):
                                       
                        print("Testing Done")
                        label_mapping = {
            0: "BG",
            1: "NC",
            2: "ED",
            3: "ET"
        }
                        for batch in range(mask_validation.shape[0]):
                            visualize_slices(mask_validation, title=f'mask_validation_Data_{batch}')
                            
                            visualize_slices(reconstruction_validation, title=f'reconstruction_validation_Data_{batch}')
                   
            total_time = time.time() - total_start
            print(f"train completed, total time: {total_time}.")
            
        else:
            print("No Training argument provided")
            

            autoencoder_WT = VQVAE_seq_WT(in_channels=1, out_channels=4, dropout_prob=0.0)

            autoencoder_WT.to(device)
            # print(model)
           
            checkpoint_dir = args.checkpoint_dir
    
            checkpoint_path = os.path.join(checkpoint_dir, 'vae_checkpoint_latest_WT.pth')
            checkpoint = torch.load(checkpoint_path)
    
    
            autoencoder_WT.load_state_dict(checkpoint['model_state_dict'], strict=False)
                  
            
            autoencoder_WT.eval()
            for param in autoencoder_WT.parameters():
                param.requires_grad = False
    
            print("DONE with VQVAE Model loading")


            autoencoder_TC = VQVAE_seq_TC(in_channels=1, out_channels=4, dropout_prob=0.0)

            autoencoder_TC.to(device)
    
            checkpoint_dir = args.checkpoint_dir
    
            checkpoint_path = os.path.join(checkpoint_dir, 'vae_checkpoint_latest_TC.pth')
            checkpoint = torch.load(checkpoint_path)
    
    
            autoencoder_TC.load_state_dict(checkpoint['model_state_dict'], strict=False)
                  
            
            autoencoder_TC.eval()
            for param in autoencoder_TC.parameters():
                param.requires_grad = False
    
            print("DONE with VQVAE Model loading")


            autoencoder_ET = VQVAE_seq_ET(in_channels=1, out_channels=4, dropout_prob=0.0)

            autoencoder_ET.to(device)
    
            checkpoint_dir = args.checkpoint_dir
    
            checkpoint_path = os.path.join(checkpoint_dir, 'vae_checkpoint_latest_ET.pth')
            checkpoint = torch.load(checkpoint_path)
    
    
            autoencoder_ET.load_state_dict(checkpoint['model_state_dict'], strict=False)
                  
            
            autoencoder_ET.eval()
            for param in autoencoder_ET.parameters():
                param.requires_grad = False
    
            print("DONE with VQVAE Model loading")
            
            
            print("Cond VQVAE model Loading")
            model_WT = UNet3DResidual_stack(autoencoder_WT.quantizer0, autoencoder_WT.conv3, autoencoder_WT.conv4, autoencoder_WT.decoder, autoencoder_WT.segmentation, in_channels=4, out_channels=1, dropout_prob=0.0)

            model_WT.to(device)
            # print(model)
            
            if torch.cuda.is_available():
                input = torch.randn(1,4, 128, 128, 155).to(device)
                flops, params = profile(model_WT, (input,))
                print('Params are = ' + str(params/1000**2) + 'M')
                print('FLOPs are = ' + str(flops/1000**3) + 'G')
    
            checkpoint_dir = args.checkpoint_dir
    
            checkpoint_path = os.path.join(checkpoint_dir, 'cond_vae_checkpoint_latest_WT.pth')
            checkpoint = torch.load(checkpoint_path)
    
            # pre_trained_embedding = checkpoint['embedding0']
    
            # print("pre_trained_embedding size is", pre_trained_embedding.shape)
    
    
            model_WT.load_state_dict(checkpoint['model_state_dict'], strict=False)
                  
            
            model_WT.eval()
            for param in model_WT.parameters():
                param.requires_grad = False
    
            print("DONE with con WT Model loading")

            print("Cond VQVAE model Loading")
            model_TC = UNet3DResidual_stack(autoencoder_TC.quantizer0, autoencoder_TC.conv3, autoencoder_TC.conv4, autoencoder_TC.decoder, autoencoder_TC.segmentation, in_channels=4, out_channels=1, dropout_prob=0.0)

            model_TC.to(device)
            # print(model)
            
            if torch.cuda.is_available():
                input = torch.randn(1,4, 128, 128, 155).to(device)
                flops, params = profile(model_TC, (input,))
                print('Params = ' + str(params/1000**2) + 'M')
                print('FLOPs = ' + str(flops/1000**3) + 'G')
    
            checkpoint_dir = args.checkpoint_dir
    
            checkpoint_path = os.path.join(checkpoint_dir, 'cond_vae_checkpoint_latest_TC.pth')
            checkpoint = torch.load(checkpoint_path)
    
            # pre_trained_embedding = checkpoint['embedding0']
    
            # print("pre_trained_embedding size is", pre_trained_embedding.shape)
    
    
            model_TC.load_state_dict(checkpoint['model_state_dict'], strict=False)
                  
            
            model_TC.eval()
            for param in model_TC.parameters():
                param.requires_grad = False
    
            print("DONE with con TC Model loading")

            print("Cond VQVAE model Loading")
            model_ET = UNet3DResidual_stack(autoencoder_ET.quantizer0, autoencoder_ET.conv3, autoencoder_ET.conv4, autoencoder_ET.decoder, autoencoder_ET.segmentation, in_channels=4, out_channels=1, dropout_prob=0.0)

            model_ET.to(device)
            # print(model)
            
            # if torch.cuda.is_available():
            #     input = torch.randn(1,1, 240, 240, 155).to(device)
            #     flops, params = profile(autoencoder, (input,))
            #     print('Params = ' + str(params/1000**2) + 'M')
            #     print('FLOPs = ' + str(flops/1000**3) + 'G')
    
            checkpoint_dir = args.checkpoint_dir
    
            checkpoint_path = os.path.join(checkpoint_dir, 'cond_vae_checkpoint_latest_ET.pth')
            checkpoint = torch.load(checkpoint_path)
    
            # pre_trained_embedding = checkpoint['embedding0']
    
            # print("pre_trained_embedding size is", pre_trained_embedding.shape)
    
    
            model_ET.load_state_dict(checkpoint['model_state_dict'], strict=False)
                  
            
            model_ET.eval()
            for param in model_ET.parameters():
                param.requires_grad = False
    
            print("DONE with con TC Model loading")

            

            def log_latent_space(latents, latent_name, epoch, n_clusters=2):
                """
                Visualizes and logs each latent space separately to W&B using GPU-based t-SNE and Seaborn scatter plot.
                
                Args:
                latents: The latent vectors extracted from a specific codebook, shape (batch_size, num_channels, height, width, depth)
                latent_name: The name to log for each quantized latent (e.g., z_quantized0)
                epoch: Current epoch to track the training phase
                n_clusters: Number of clusters for coloring the latent points
                """
                # Reshape latents to (batch_size * spatial dimensions, num_channels)
                latents_reshaped = latents.view(-1, latents.size(1))  # Shape (batch_size * 30 * 30 * 20, num_channels)
            
                # Detach from computation graph before converting to NumPy
                latents_reshaped = latents_reshaped.detach().cpu().numpy()
            
                # Check the number of samples in reshaped latents
                num_samples = latents_reshaped.shape[0]
            
                # Set the perplexity, ensure it is smaller than the number of samples
                perplexity = min(30, num_samples - 1)
            
                # Apply GPU-based t-SNE to reduce the high-dimensional latent space to 2D
                tsne = cuml.TSNE(n_components=2, perplexity=perplexity)
            
                try:
                    latents_2d = tsne.fit_transform(latents_reshaped)
            
                    # Apply KMeans to find clusters in the latent space
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                    cluster_labels = kmeans.fit_predict(latents_reshaped)
            
                    # Create a DataFrame for easier plotting
                    df = pd.DataFrame(latents_2d, columns=['x', 'y'])
                    df['cluster'] = cluster_labels
            
                    # Create the scatter plot using Seaborn
                    plt.figure(figsize=(10, 8))
                    sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='viridis', style='cluster', markers='o', s=100)
                    plt.title(f"{latent_name} Latent Space at Epoch {epoch}")
                    plt.xlabel('TSNE Component 1')
                    plt.ylabel('TSNE Component 2')
            
                    # Save the plot to a temporary file
                    plot_path = f"{latent_name}_latent_space_epoch.png"
                    plt.savefig(plot_path)
                    plt.close()
            
                    # Log the image to W&B
                    wandb.log({f"{latent_name}_latent_space": wandb.Image(plot_path)})
            
                except ValueError as e:
                    print(f"Error visualizing latent space for {latent_name}: {e}")
                    
            
            mask_validation, reconstruction_validation, t2f = test_cond(model_WT, autoencoder_WT, model_TC, autoencoder_TC, model_ET, autoencoder_ET, test_loader, len(test_dataset), device)
                

if __name__ == "__main__":
    args = get_args()  # Import arguments from the config file
    print("args:", args)  # Print arguments for verification
    run_pipeline(args)  # Run the data pipeline
