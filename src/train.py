from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.nn import L1Loss
import visdom
import nibabel as nib
import numpy as np
import os
from monai.utils import first, set_determinism
from torch.optim import Adam


import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.losses.focal_loss import FocalLoss
from monai.losses.spatial_mask import MaskedLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, deprecated_arg, look_up_option, pytorch_after
from monai.metrics import compute_hausdorff_distance

mse_loss = L1Loss()
class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.NONE,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        weight = torch.as_tensor(weight) if weight is not None else None
        self.register_buffer("class_weight", weight)
        self.class_weight: None | torch.Tensor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> from monai.losses.dice import *  # NOQA
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                # print("target shape is", target.shape)
                target = one_hot(target, num_classes=n_pred_ch)
                # print("target shape is", target.shape)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
                print("target shape is", target.shape)
                print("input shape is", input.shape)

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o
        # print(f"Intersection: {intersection}")
        # print(f"Denominator: {denominator}")

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        dice = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        # print(f"Dice: {dice}")

        num_of_classes = target.shape[1]
        if self.class_weight is not None and num_of_classes != 1:
            # make sure the lengths of weights are equal to the number of classes
            if self.class_weight.ndim == 0:
                self.class_weight = torch.as_tensor([self.class_weight] * num_of_classes)
            else:
                if self.class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            if self.class_weight.min() < 0:
                raise ValueError("the value/values of the `weight` should be no less than 0.")
            # apply class_weight to loss
            f = f * self.class_weight.to(f)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f
# from losses import DiceCELoss
# from monai.metrics import DiceMetric, compute_meandice, compute_hausdorff_distance, compute_average_surface_distance
weight_BG = 1.0   # Weight for Edema class
weight_ED = 1.0   # Weight for Edema class
weight_NC = 2.0   # Weight for Necrotic Core class (higher because it's underperforming)
weight_ET = 2.0   # Weight for Enhancing Tumor class (higher because it's underperforming)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = torch.tensor([weight_BG, weight_NC, weight_ED, weight_ET], dtype=torch.float32).to(device)
dice_loss = DiceLoss(to_onehot_y=False, softmax=False)
dice_loss2 = DiceLoss(to_onehot_y=False, softmax=False)
# dice_loss = DiceLoss(to_onehot_y=True, softmax=False)


scaler = GradScaler()



def train_vae(model, train_loader, train_dataset_len, optimizer, device):
    """
    Train the VAE model for one epoch with mixed precision.
    """
    model = model.to(device)
    #for epoch in range(n_epochs):
    model.train()
    scaler = GradScaler()
    epoch_loss = 0
    quantization_losses = 0
    # class_names = ["TC", "WT", "ET"]  # Names of the classes
    # Initialize dictionary to store sum of normalized losses
    class_losses_sum_overall_wo = {'BG': 0, 'ET': 0}
    class_losses_sum_overall = {'ET': 0}
    # class_losses_sum_overall = {"BG":0, 'NC': 0}
    batch_count = 0
    encodings_sumation = torch.zeros(512).to(device)
    torch.autograd.set_detect_anomaly(True)
    for step, batch in enumerate(train_loader):
    
        if 'mask' in batch:
            mask = batch['mask']
            # print("image shape with seg_mask is", mask.shape)
        else:
            raise KeyError("Key 'segmentation' not found in batch_data") 
        
        optimizer.zero_grad(set_to_none=True)
        # images = images.to(device)
        mask = mask.to(device)
        mask_up = mask[:, 1:2, :, :, :]
        mask = mask[:, 0:2, :, :, :]
       
        with autocast(device_type='cuda', enabled=False):
            
    
           
            z_quantized_all, reconstruction, quantization_loss, encodings_sum0, embedding0 = model(mask_up)
            
            
            non_zero_count = torch.count_nonzero(encodings_sum0)
            print(f"Number of non-zero elements: {non_zero_count}")
            
            encodings_sumation += encodings_sum0

            quantization_loss = quantization_loss
            
            
    
            combined_loss = dice_loss(reconstruction, mask)
            # print("combined_loss shape is", combined_loss.shape)
            combined_loss = combined_loss.mean(dim=0)

            # print(f"ET_loss_{combined_loss[0]}")
            print(f"BG_loss_{combined_loss[0]}_____________ET_loss_{combined_loss[1]}")

            loss_BG = combined_loss[0]
            
            loss_EN = combined_loss[1]
           
            
            
            re_norm_combined_loss = ((loss_EN+loss_BG))
            print("re_norm_combined_loss", re_norm_combined_loss)

            

           
            print("quantization_losses is", quantization_loss)
            batch_images = batch['mask'].shape[0]


            for idx, (key, value) in enumerate(class_losses_sum_overall_wo.items()):
                class_losses_sum_overall_wo[key]+=((combined_loss[idx].item())*batch_images)

            
           
            
    
            loss = (0,75*re_norm_combined_loss + quantization_loss)
            print("total loss is", loss / 3)
            loss_tr = loss*batch_images
            quantization_loss=quantization_loss.item()*batch_images
    
        scaler.scale(loss).backward()  # Scale loss and perform backward pass
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        scaler.step(optimizer)  # Update model parameters
        scaler.update()
    
    
        epoch_loss += loss_tr.item()
        quantization_losses += quantization_loss

    for key, value in class_losses_sum_overall_wo.items():
        class_losses_sum_overall_wo[key] = value / train_dataset_len
    
    for key, value in class_losses_sum_overall.items():
        class_losses_sum_overall[key] = value / train_dataset_len
    # Return the average loss over the epoch
    return epoch_loss / train_dataset_len, class_losses_sum_overall, class_losses_sum_overall_wo


def validate_vae(model, model_inferer, dataloader, val_dataset_len, device):
    """
    Validate the VAE model on the validation dataset.
    """
    print("Validation in Progress")
    # model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    val_loss = 0  # Initialize total loss accumulator
    quantization_losses = 0
    class_losses_sum_overall_wo = {'BG': 0, 'ET': 0}
    class_losses_sum_overall = {'ET': 0}
    # class_losses_sum_overall = {'BG': 0, 'NC': 0}
    with torch.no_grad():  # Disable gradient computation for validation
        
        for val_step, batch in enumerate(dataloader, start=1):
            
                       
            
            if 'mask' in batch:
                mask = batch['mask']
                # print("image shape with seg_mask is", mask.shape)
            else:
                raise KeyError("Key 'segmentation' not found in batch_data") 

            # images = images.to(device)
            mask = mask.to(device)
           
            mask_up = mask[:,1:2,:,:,:]

            mask = mask[:, 0:2, :, :, :]
            
            with autocast(device_type='cuda', enabled=False):  # Mixed precision context for validation
                
                z_quantized_all, reconstruction, quantization_loss, encodings_sum0, embedding0 = model(mask_up)
                if reconstruction.shape[4] > 155:
                    # print("256.shape", reconstruction.shape)
                    reconstruction = reconstruction[:, :, :, :, :-1]
                    
                combined_loss = dice_loss(reconstruction, mask)
                # print("combined_loss shape is", combined_loss.shape)
                combined_loss = combined_loss.mean(dim=0)
    
                # print(f"ET_loss_{combined_loss[0]}")
                print(f"BG_loss_{combined_loss[0]}______ET_loss_{combined_loss[1]}")
                # print("combined_loss shape is", combined_loss.shape)
                # print("combined_loss is", combined_loss)
                loss_BG = combined_loss[0]
               
                loss_EN = combined_loss[1]
                
                quantization_loss = quantization_loss
                re_norm_combined_loss = ((loss_BG+loss_EN))
                
                batch_images = batch['mask'].shape[0]

               

                for idx, (key, value) in enumerate(class_losses_sum_overall_wo.items()):
                    class_losses_sum_overall_wo[key]+=((combined_loss[idx].item())*batch_images)

              
                loss = (0,75*re_norm_combined_loss + quantization_loss)
                print("total loss is", loss / 3)
                loss_val = loss*batch_images
                quantization_loss=quantization_loss.item()*batch_images
                
    

    
            
            val_loss += loss_val.item()  # Accumulate the loss value
            quantization_losses += quantization_loss

    for key, value in class_losses_sum_overall_wo.items():
        class_losses_sum_overall_wo[key] = value / val_dataset_len

    
    for key, value in class_losses_sum_overall.items():
        class_losses_sum_overall[key] = value / val_dataset_len

    # Return the average loss over the validation dataset
    return val_loss / val_dataset_len, class_losses_sum_overall, class_losses_sum_overall_wo





def test_vae(model_WT, model_TC, model_ET, model_inferer, dataloader, val_dataset_len, device):
    """
    Validate the VAE model on the validation dataset.
    """
    print("Validation in Progress")
    # model = model.to(device)
    model_WT.eval()  # Set the model to evaluation mode
    model_TC.eval()
    model_ET.eval()
    val_loss = 0  # Initialize total loss accumulator
    quantization_losses = 0
    class_losses_sum_overall_wo = {'WT': 0, 'TC': 0, 'ET': 0}
    class_losses_sum_overall = {'BG': 0, 'TC': 0, 'WT': 0, 'ET': 0}
    hd_95_app = {"WT_HD_95": 0, "TC_HD_95": 0, "ET_HD_95": 0}
    count = 0
    def save_average():
        if count > 0:
            avg_hd_95 = {key: hd_95_app[key] / count for key in hd_95_app}  # Compute average
            print("Final Averaged HD95:", avg_hd_95)
            return avg_hd_95
        else:
            print("No valid updates to compute average")
            return None
    # class_losses_sum_overall = {'BG': 0, 'NC': 0}
    hd_95npes = []
    with torch.no_grad():  # Disable gradient computation for validation
        
        for batch in tqdm(dataloader, desc='inference'):
            
                       
            images={}
            for key in ["t1n", "t2w", "t1c", "t2f"]:
                if key in batch:
                    images[key] = batch[key]
                    #print(f"image shape with modality {key} is", batch[key].shape)
                else:
                    raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
        
            # Stack modalities along the channel dimension (dim=1)
            images = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1)
            # print("image shape with stacked modality is", images.shape)
            t2f = batch['t2f_normalized']
            # Get the segmentation mask from batch_data
            if 'mask' in batch:
                mask = batch['mask']
                # print("image shape with seg_mask is", mask.shape)
            else:
                raise KeyError("Key 'segmentation' not found in batch_data") 
            images = images.to(device)
            
            mask = mask.to(device)
            mask_up_WT = mask[:,1:2,:,:,:]
            mask_up_TC = mask[:,2:3,:,:,:]
            mask_up_ET = mask[:,3:,:,:,:]
            
            with autocast(device_type='cuda', enabled=False):  # Mixed precision context for validation
                
                reconstruction_WT = model_WT(mask_up_WT)
                reconstruction_WT_mask = torch.argmax(reconstruction_WT, dim=1)
                reconstruction_ET = reconstruction_WT[:,1:,:,:,:]
                reconstruction_TC = model_TC(mask_up_TC)
                reconstruction_TC_mask = torch.argmax(reconstruction_TC, dim=1)
                reconstruction_TC = reconstruction_TC[:,1:,:,:,:]
                reconstruction_ET = model_ET(mask_up_ET)
                reconstruction_ET_mask = torch.argmax(reconstruction_ET, dim=1)
                reconstruction_ET = reconstruction_ET[:,1:,:,:,:]
                reconstruction = torch.cat((reconstruction_WT, reconstruction_TC, reconstruction_ET), dim=1)
                reconstruction_mask = torch.stack((reconstruction_WT_mask, reconstruction_TC_mask, reconstruction_ET_mask), dim=1)
                print("reconstruction_WT_mask", torch.sum(reconstruction_WT_mask==1))
                print("mask_up_WT", torch.sum(mask_up_WT==1))
                print("reconstruction_TC_mask", torch.sum(reconstruction_TC_mask==1))
                print("mask_up_TC", torch.sum(mask_up_TC==1))
                print("reconstruction_ET_mask", torch.sum(reconstruction_ET_mask==1))
                print("mask_up_ET", torch.sum(mask_up_ET==1))
                mask = torch.cat((mask_up_WT, mask_up_TC, mask_up_ET), dim=1)
                

                # reconstruction_mask = torch.stack((reconstruction_WT_mask, reconstruction_TC_mask, reconstruction_ET_mask), dim=1)  # Shape: [B, 3, D, H, W]

                # Compute Edema (ED) and Necrotic Core (NC)
                edema_mask = reconstruction_WT_mask.squeeze(1) - reconstruction_TC_mask.squeeze(1)  # ED = WT - TC
                necrotic_core_mask = reconstruction_TC_mask.squeeze(1) - reconstruction_ET_mask.squeeze(1)  # NC = TC - ET
                
                # Compute Background (BG) → Pixels that are 0 in all three masks
                background_mask = (reconstruction_WT_mask.squeeze(1) == 0) & (reconstruction_TC_mask.squeeze(1) == 0) & (reconstruction_ET_mask.squeeze(1) == 0)  
                background_mask = background_mask.to(reconstruction_WT_mask.dtype)  # Convert to same dtype as masks
                
                # Stack all masks in order: [BG, ED, NC, ET]
                final_mask_stack = torch.stack((background_mask, edema_mask, necrotic_core_mask, reconstruction_ET_mask), dim=1)  # Shape: [B, 4, D, H, W]
                print("final_mask_stack", final_mask_stack.shape)
                
                # Apply argmax along the channel dimension to get final segmentation mask
                final_segmentation_mask = torch.argmax(final_mask_stack, dim=1)  # Shape: [B, D, H, W]


                WT_gt, TC_gt, ET_gt = mask_up_WT.squeeze(1), mask_up_TC.squeeze(1), mask_up_ET.squeeze(1)  # Shape: [B, D, H, W] each
                
                # Compute Edema (ED) and Necrotic Core (NC)
                ED_gt = WT_gt - TC_gt  # Edema = WT - TC
                NC_gt = TC_gt - ET_gt  # Necrotic Core = TC - ET
                
                # Compute Background (BG) → Pixels that are 0 in all three masks
                BG_gt = (WT_gt == 0) & (TC_gt == 0) & (ET_gt == 0)
                BG_gt = BG_gt.to(WT_gt.dtype)  # Convert to same dtype as masks
                
                # Stack all masks in order: [BG, ED, NC, ET]
                final_GT_stack = torch.stack((BG_gt, ED_gt, NC_gt, ET_gt), dim=1)  # Shape: [B, 4, D, H, W]
                print("final_GT_stack", final_GT_stack.shape)
                
                # Apply argmax along the channel dimension to get final GT segmentation mask
                final_GT_segmentation_mask = torch.argmax(final_GT_stack, dim=1)  # Shape: [B, D, H, W]


                


                combined_loss = dice_loss(reconstruction_mask, mask)
                # print("combined_loss shape is", combined_loss.shape)
                combined_loss = combined_loss.mean(dim=0)


                # print(f"BG_loss_{combined_loss[0]}_____________NC_loss_{combined_loss[1]}___________ED_loss_{combined_loss[2]}_____________ET_loss_{combined_loss[3]}")
                # print("val_step", val_step)
                print(f"WT_loss_{combined_loss[0]}__________TC_loss_{combined_loss[1]}_____________ET_loss_{combined_loss[2]}")
                # print("val_step", val_step)
                loss_BG = combined_loss[0]
                
                loss_EN = combined_loss[1]
               
                loss = loss_BG+loss_EN
               
                batch_images = batch['mask'].shape[0]
                
                loss = loss*batch_images
                # indices_loss = indices_loss*batch_images
                # loss_val = loss*batch_images
                
                
                # quantization_loss=quantization_loss.item()*batch_images
                print("Mask unique values:", mask.shape)
                print("Reconstruction mask unique values:", reconstruction_mask.shape)
                hd_wt = compute_hausdorff_distance(reconstruction_mask[:, 0:1, :, :, :], mask[:, 0:1, :, :, :], percentile=95)
                hd_tc = compute_hausdorff_distance(reconstruction_mask[:, 1:2, :, :, :], mask[:, 1:2, :, :, :], percentile=95)
                hd_et = compute_hausdorff_distance(reconstruction_mask[:, 2:3, :, :, :], mask[:, 2:3, :, :, :], percentile=95)
                
                hd_95 = {"WT_HD_95": hd_wt.item(), "TC_HD_95": hd_tc.item(), "ET_HD_95": hd_et.item()}
                hd95_np = np.array(list(hd_95.values()))

               
                if np.all(np.isfinite(hd95_np)):
                    hd_95_app = {key: hd_95_app[key] + hd_95[key] for key in hd_95_app}
                    count += 1  # Increment counter
                    print("Updated hd_95_app:", hd_95_app)
                    for idx, (key, value) in enumerate(class_losses_sum_overall_wo.items()):
                        class_losses_sum_overall_wo[key]+=(1-(combined_loss[idx].item())*batch_images)
                else:
                    print("Skipping NaN or Inf HD95 value")
                
                    
            
           
    for key, value in class_losses_sum_overall_wo.items():
        class_losses_sum_overall_wo[key] = value / count
    print("count is", count)
    print("class_losses_sum_overall_wo", class_losses_sum_overall_wo)
    hd_95 = save_average()
    print("hd_95", hd_95)
   
    return final_GT_segmentation_mask, final_segmentation_mask, t2f, class_losses_sum_overall_wo, hd_95
            # mask = torch.argmax(mask, dim=1)
            # reconstruction = torch.argmax(reconstruction, dim=1) 
            # if combined_loss[1].item()>=0.2:
            # yield final_GT_segmentation_mask, final_segmentation_mask, t2f, class_losses_sum_overall_wo, hd_95
