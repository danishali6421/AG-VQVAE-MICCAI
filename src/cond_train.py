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

# from monai.losses.focal_loss import FocalLoss
from monai.losses.spatial_mask import MaskedLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, deprecated_arg, look_up_option, pytorch_after
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import NormalizeIntensity
from sklearn.preprocessing import StandardScaler
import torchmetrics
from medpy.metric.binary import hd95
from monai.metrics import compute_hausdorff_distance
from monai.losses import SSIMLoss
# Initialize the SSIM metric
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssim_loss = (torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=(7, 7, 7))).to(device)

# Initialize and apply normalization
normalize_intensity = NormalizeIntensity()


norm_scale = StandardScaler()


# Create a SummaryWriter to log to the "runs" directory
writer = SummaryWriter(log_dir='runs')

# During training, log the gradients and other metrics
mse_loss = nn.MSELoss()#L1Loss()#nn.MSELoss()#L1Loss()
ce_loss2 = nn.CrossEntropyLoss(ignore_index=-1)
ce_loss = nn.CrossEntropyLoss()   


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_mri, z_mask):
        """
        z_mri: (batch, channels, height, width, depth) - MRI latent representations
        z_mask: (batch, channels, height, width, depth) - Corresponding segmentation mask latents
        negatives: (batch, num_negatives, channels, height, width, depth) - Negative samples

        Returns:
        Contrastive loss value
        """
        batch_size = z_mri.shape[0]
        negatives = torch.randn(batch_size, 5, 32, 60, 60, 38).to(device)  # 5 random negative MRI latents

        # Flatten the spatial dimensions (channels, height, width, depth) into a single vector
        z_mri_flat = z_mri.view(batch_size, -1)  # Shape: (batch, channels * height * width * depth)
        z_mask_flat = z_mask.view(batch_size, -1)  # Shape: (batch, channels * height * width * depth)
        negatives_flat = negatives.view(batch_size, negatives.shape[1], -1)  # Shape: (batch, num_negatives, channels * height * width * depth)

        # Compute cosine similarity
        pos_sim = F.cosine_similarity(z_mri_flat, z_mask_flat, dim=-1)  # Positive pairs
        neg_sim = F.cosine_similarity(z_mri_flat.unsqueeze(1), negatives_flat, dim=-1)  # Negative pairs

        # Compute softmax over positives and negatives
        pos_exp = torch.exp(pos_sim / self.temperature)
        neg_exp = torch.exp(neg_sim / self.temperature).sum(dim=1)  # Sum over negatives

        # Compute contrastive loss
        loss = -torch.log(pos_exp / (pos_exp + neg_exp))
        return loss.mean()

loss_fn = ContrastiveLoss(temperature=0.1)



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
                # print("target shape is", target.shape)
                # print("input shape is", input.shape)

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


class FocalLoss(_Loss):
    """
    FocalLoss is an extension of BCEWithLogitsLoss that down-weights loss from
    high confidence correct predictions.

    Reimplementation of the Focal Loss described in:

        - ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002), T. Lin et al., ICCV 2017
        - "AnatomyNet: Deep learning for fast and fully automated whole-volume segmentation of head and neck anatomy",
          Zhu et al., Medical Physics 2018

    Example:
        >>> import torch
        >>> from monai.losses import FocalLoss
        >>> from torch.nn import BCEWithLogitsLoss
        >>> shape = B, N, *DIMS = 2, 3, 5, 7, 11
        >>> input = torch.rand(*shape)
        >>> target = torch.rand(*shape)
        >>> # Demonstrate equivalence to BCE when gamma=0
        >>> fl_g0_criterion = FocalLoss(reduction='none', gamma=0)
        >>> fl_g0_loss = fl_g0_criterion(input, target)
        >>> bce_criterion = BCEWithLogitsLoss(reduction='none')
        >>> bce_loss = bce_criterion(input, target)
        >>> assert torch.allclose(fl_g0_loss, bce_loss)
        >>> # Demonstrate "focus" by setting gamma > 0.
        >>> fl_g2_criterion = FocalLoss(reduction='none', gamma=2)
        >>> fl_g2_loss = fl_g2_criterion(input, target)
        >>> # Mark easy and hard cases
        >>> is_easy = (target > 0.7) & (input > 0.7)
        >>> is_hard = (target > 0.7) & (input < 0.3)
        >>> easy_loss_g0 = fl_g0_loss[is_easy].mean()
        >>> hard_loss_g0 = fl_g0_loss[is_hard].mean()
        >>> easy_loss_g2 = fl_g2_loss[is_easy].mean()
        >>> hard_loss_g2 = fl_g2_loss[is_hard].mean()
        >>> # Gamma > 0 causes the loss function to "focus" on the hard
        >>> # cases.  IE, easy cases are downweighted, so hard cases
        >>> # receive a higher proportion of the loss.
        >>> hard_to_easy_ratio_g2 = hard_loss_g2 / easy_loss_g2
        >>> hard_to_easy_ratio_g0 = hard_loss_g0 / easy_loss_g0
        >>> assert hard_to_easy_ratio_g2 > hard_to_easy_ratio_g0
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        gamma: float = 2.0,
        alpha: float | None = None,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        use_softmax: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the loss calculation.
                If False, `alpha` is invalid when using softmax.
            to_onehot_y: whether to convert the label `y` into the one-hot format. Defaults to False.
            gamma: value of the exponent gamma in the definition of the Focal loss. Defaults to 2.
            alpha: value of the alpha in the definition of the alpha-balanced Focal loss.
                The value should be in [0, 1]. Defaults to None.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            use_softmax: whether to use softmax to transform the original logits into probabilities.
                If True, softmax is used. If False, sigmoid is used. Defaults to False.

        Example:
            >>> import torch
            >>> from monai.losses import FocalLoss
            >>> pred = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)
            >>> grnd = torch.tensor([[0], [1], [0]], dtype=torch.int64)
            >>> fl = FocalLoss(to_onehot_y=True)
            >>> fl(pred, grnd)
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.use_softmax = use_softmax
        weight = torch.as_tensor(weight) if weight is not None else None
        self.register_buffer("class_weight", weight)
        self.class_weight: None | torch.Tensor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                The input should be the original logits since it will be transformed by
                a sigmoid/softmax in the forward function.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            ValueError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
            ValueError: When ``self.weight`` is a sequence and the length is not equal to the
                number of classes.
            ValueError: When ``self.weight`` is/contains a value that is less than 0.

        """
        n_pred_ch = input.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise ValueError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        loss: Optional[torch.Tensor] = None
        input = input.float()
        target = target.float()
        if self.use_softmax:
            if not self.include_background and self.alpha is not None:
                self.alpha = None
                warnings.warn("`include_background=False`, `alpha` ignored when using softmax.")
            loss = softmax_focal_loss(input, target, self.gamma, self.alpha)
        else:
            loss = sigmoid_focal_loss(input, target, self.gamma, self.alpha)

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
            self.class_weight = self.class_weight.to(loss)
            broadcast_dims = [-1] + [1] * len(target.shape[2:])
            self.class_weight = self.class_weight.view(broadcast_dims)
            loss = self.class_weight * loss

        if self.reduction == LossReduction.SUM.value:
            # Previously there was a mean over the last dimension, which did not
            # return a compatible BCE loss. To maintain backwards compatible
            # behavior we have a flag that performs this extra step, disable or
            # parameterize if necessary. (Or justify why the mean should be there)
            average_spatial_dims = True
            if average_spatial_dims:
                loss = loss.mean(dim=list(range(2, len(target.shape))))
            loss = loss.sum()
        elif self.reduction == LossReduction.MEAN.value:
            loss = loss.mean()
        elif self.reduction == LossReduction.NONE.value:
            pass
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return loss


def softmax_focal_loss(
    input: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: Optional[float] = None
) -> torch.Tensor:
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
    s_j is the unnormalized score for class j.
    """
    input_ls = input.log_softmax(1)
    loss: torch.Tensor = -(1 - input_ls.exp()).pow(gamma) * input_ls * target

    if alpha is not None:
        # (1-alpha) for the background class and alpha for the other classes
        alpha_fac = torch.tensor([1 - alpha] + [alpha] * (target.shape[1] - 1)).to(loss)
        broadcast_dims = [-1] + [1] * len(target.shape[2:])
        alpha_fac = alpha_fac.view(broadcast_dims)
        loss = alpha_fac * loss

    return loss


def sigmoid_focal_loss(
    input: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: Optional[float] = None
) -> torch.Tensor:
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p = sigmoid(x), pt = p if label is 1 or 1 - p if label is 0
    """
    # computing binary cross entropy with logits
    # equivalent to F.binary_cross_entropy_with_logits(input, target, reduction='none')
    # see also https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp#L363
    loss: torch.Tensor = input - input * target - F.logsigmoid(input)

    # sigmoid(-i) if t==1; sigmoid(i) if t==0 <=>
    # 1-sigmoid(i) if t==1; sigmoid(i) if t==0 <=>
    # 1-p if t==1; p if t==0 <=>
    # pfac, that is, the term (1 - pt)
    invprobs = F.logsigmoid(-input * (target * 2 - 1))  # reduced chance of overflow
    # (pfac.log() * gamma).exp() <=>
    # pfac.log().exp() ^ gamma <=>
    # pfac ^ gamma
    loss = (invprobs * gamma).exp() * loss

    if alpha is not None:
        # alpha if t==1; (1-alpha) if t==0
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_factor * loss

    return loss


class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, latent1, latent2):
        """
        Compute the cosine similarity loss between two latent tensors.
        
        Args:
            latent1: Tensor of shape (batch_size, channels, height, width, depth)
            latent2: Tensor of shape (batch_size, channels, height, width, depth)
        
        Returns:
            loss: The computed cosine similarity loss
        """
        # Reshape both latents into 2D tensors: (batch_size, feature_dim)
        latent1_flat = latent1.view(latent1.size(0), -1)  # Shape: (batch_size, feature_dim)
        latent2_flat = latent2.view(latent2.size(0), -1)  # Shape: (batch_size, feature_dim)

        # Normalize the latents (i.e., divide by their L2 norm)
        latent1_normalized = F.normalize(latent1_flat, p=2, dim=1)  # Normalize along the feature dimension
        latent2_normalized = F.normalize(latent2_flat, p=2, dim=1)

        # Compute cosine similarity
        cosine_similarity = torch.sum(latent1_normalized * latent2_normalized, dim=1)  # Dot product

        # Compute cosine similarity loss (1 - cosine similarity)
        loss = 1 - cosine_similarity.mean()  # Mean over the batch

        return loss


cosine_loss = CosineSimilarityLoss()


criterion = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean", to_onehot_y=True, use_softmax=True)

dice_loss = DiceLoss(to_onehot_y=False, softmax=False)

mse_loss = nn.MSELoss()



scaler = GradScaler()

class DynamicAttentionMasking:
    def __init__(self, num_classes, smoothing=0.01):
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.index_counts = torch.zeros(num_classes, dtype=torch.float32).to(device)

    def update_counts(self, predictions):
        # Update counts based on predictions (logits to class indices)
        _, predicted_indices = torch.max(predictions, dim=-1)
        unique, counts = torch.unique(predicted_indices, return_counts=True)
        self.index_counts[unique] += counts.float()

    def compute_mask(self):
        # Compute utilization scores (normalized counts)
        utilization_scores = self.index_counts / self.index_counts.sum()

        # Invert utilization to create the mask (higher for under-utilized)
        mask = 1.0 / (utilization_scores + self.smoothing)
        mask /= mask.sum()  # Normalize the mask
        return mask






def train_cond(model, autoencoder, train_loader, train_dataset_len, optimizer, device):
    """
    Train the VAE model for one epoch with mixed precision.
    """
    model = model.to(device)
    #for epoch in range(n_epochs):
    model.train()
    # scaler = GradScaler()
    epoch_loss = 0
    quantization_losses = 0
    q_losses = 0
    mse_losses = 0
    latent_losses = 0
    indices_losses = 0
    # class_names = ["TC", "WT", "ET"]  # Names of the classes
    # Initialize dictionary to store sum of normalized losses
    class_losses_sum_overall_wo = {"BG":0, 'ET': 0}
    class_losses_sum_overall = {"BG":0, 'TC': 0, 'WT': 0, 'ET': 0}
    # class_losses_sum_overall = {"BG":0, 'NC': 0}
    batch_count = 0
    encodings_sumation = torch.zeros(64).to(device)
    dynamic_masking = DynamicAttentionMasking(num_classes=128)
    torch.autograd.set_detect_anomaly(True)
    for step, batch in enumerate(train_loader):
        # print("batch_count", batch_count)
        # batch_count += 1
        # print("step", step)
        # print("length of train loader", len(train_loader))
        with torch.no_grad():   
            # with autocast(device_type='cuda', enabled=True):
            print("Training in Progress")
            # mask = batch['mask'].to(device)
            images={}
            for key in ["t1n", "t2w", "t1c", "t2f"]:
                if key in batch:
                    images[key] = (batch[key])
                    #print(f"image shape with modality {key} is", batch[key].shape)
                else:
                    raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
            # random_number = torch.randint(0, 1, (1,)).item()
            # if random_number == 0:
            images = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1)
            # print("images after stacking is of shape", images.shape)
            # Get the segmentation mask from batch_data
            if 'mask' in batch:
                mask = batch['mask']
                # print("image shape with seg_mask is", mask.shape)
            else:
                raise KeyError("Key 'segmentation' not found in batch_data") 
            optimizer.zero_grad(set_to_none=True)
            mask = mask.to(device)
            # print("mask shape is", mask.shape)
            images = images.to(device)
            mask_up = mask[:, 1:2, :, :, :]
            
            
            # images = mask_cat*images
            # print("images", images.shape)
            autoencoder_latent=autoencoder.encoder(mask_up) 
            autoencoder_latent = autoencoder.bottleneck(autoencoder_latent)
            autoencoder_latent=autoencoder.conv1(autoencoder_latent) 
            autoencoder_latent=autoencoder.conv2(autoencoder_latent) 
            # autoencoder_latent=autoencoder.conv3(autoencoder_latent)
            autoencoder_latent_indices=autoencoder.quantizer0.quantize(autoencoder_latent)
            autoencoder_latent_indices_embeddingsss = autoencoder.quantizer0.embed(autoencoder_latent_indices)
            autoencoder_latent_indices = autoencoder_latent_indices.long()
            
        with autocast(device_type='cuda', enabled=False):

            
            # x_bot, x_bottt, quantized, quantized_loss, mean, std, autoencoder_latent_mean, autoencoder_latent_std = model(images, autoencoder_latent)
            x_bot, x_bottt, quantized, quantized_loss = model(images, is_train=True)
            

            

            cross_ent_loss = criterion(x_bot, torch.unsqueeze(autoencoder_latent_indices, dim=1))
            cross_ent_loss2 = ce_loss(x_bot, autoencoder_latent_indices)
            loss = 1000*cross_ent_loss + cross_ent_loss2 
            x_bot = torch.argmax(x_bot, dim=1)
            print("cross_ent_loss1", cross_ent_loss)
            print("cross_ent_loss2", cross_ent_loss2)
            # print("autoencoder_latent_indices_embeddingsss_loss", autoencoder_latent_indices_embeddingsss_loss)
            print("len where equal", torch.sum(x_bot == autoencoder_latent_indices))
            batch_images = batch['mask'].shape[0]

            cross_ent_loss = loss*batch_images
            
    
        scaler.scale(loss).backward()  # Scale loss and perform backward pass
        
        scaler.step(optimizer)  # Update model parameters
        scaler.update()
    
    
        
        latent_losses += cross_ent_loss.item()
        
    for key, value in class_losses_sum_overall_wo.items():
        class_losses_sum_overall_wo[key] = value / train_dataset_len
    return latent_losses / train_dataset_len, class_losses_sum_overall_wo




def validate_cond(model, autoencoder, dataloader, val_dataset_len, device):
    """
    Validate the VAE model on the validation dataset.
    """
    print("Validation in Progress")
    # model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    val_loss = 0  # Initialize total loss accumulator
    q_losses = 0
    mse_losses = 0
    quantization_losses = 0
    latent_losses = 0
    indices_losses = 0
    class_losses_sum_overall_wo = {'BG': 0, 'ET': 0}
    class_losses_sum_overall = {'BG': 0, 'TC': 0, 'WT': 0, 'ET': 0}
    # class_losses_sum_overall = {'BG': 0, 'NC': 0}
    hd_95es = 0
    csoine_sim_losses = 0
    counter = 0
    val_steps = []
    hd_95npes = []
    with torch.no_grad():  # Disable gradient computation for validation
        
        for val_step, batch in enumerate(dataloader):

            print("first val step", val_step)
            
                       
            images={}
            for key in ["t1n", "t2w", "t1c", "t2f"]:
                if key in batch:
                    images[key] = (batch[key])
                   # print(f"image shape with modality {key} is", batch[key].shape)
                else:
                    raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
        
            # # Stack modalities along the channel dimension (dim=1)
            # random_number = torch.randint(0, 1, (1,)).item()
            # if random_number == 0:
            images = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1)
            print("image shape with stacked modality is", images.shape)
            # images = batch['t2f']
            # Get the segmentation mask from batch_data
            if 'mask' in batch:
                mask = batch['mask']
                # print("image shape with seg_mask is", mask.shape)
            else:
                raise KeyError("Key 'segmentation' not found in batch_data") 

            mask = mask.to(device)
            mask_up = mask[:,1:2,:,:,:]
            mask = mask[:,0:2,:,:,:]
           
            images = images.to(device)
            
            autoencoder_latent=autoencoder.encoder(mask_up) 
            autoencoder_latent = autoencoder.bottleneck(autoencoder_latent)
            autoencoder_latent=autoencoder.conv1(autoencoder_latent) 
            autoencoder_latent=autoencoder.conv2(autoencoder_latent) 
            # autoencoder_latent=autoencoder.conv3(autoencoder_latent) 
            autoencoder_latent_indices=autoencoder.quantizer0.quantize(autoencoder_latent)
            autoencoder_latent_indices_embeddingsss = autoencoder.quantizer0.embed(autoencoder_latent_indices)
            autoencoder_latent_indices = autoencoder_latent_indices.long()
            
            with autocast(device_type='cuda', enabled=False):                 
               
                x_bot, x_bottt, quantized, quantized_loss =  model(images, is_train=False)
                
                x_bot = torch.argmax(x_bot, dim=1)
                
                embeddingsss = autoencoder.quantizer0.embed(x_bot)
                
                # Decoder path with skip connections
                reconstruction = autoencoder.conv3(embeddingsss)
                reconstruction = autoencoder.conv4(reconstruction)
                # reconstruction = autoencoder.conv6(reconstruction)
                reconstruction = autoencoder.decoder(reconstruction)
                reconstruction = autoencoder.segmentation(reconstruction)

                
                combined_loss = dice_loss(reconstruction, mask)
                
                combined_loss = combined_loss.mean(dim=0)
                reconstructiohd = torch.argmax(reconstruction, dim=1)
                reconstructiohd0 = (reconstructiohd == 0)
                reconstructiohd1 = (reconstructiohd == 1)
                reconstructiohd_cal = torch.unsqueeze((reconstructiohd1), dim=1)
                hd_95 = compute_hausdorff_distance(reconstructiohd_cal, mask_up, percentile=95)
                hd_95 = hd_95.mean(dim=0)
                print("Hausdorff Distance (95th percentile):", hd_95)

                # print(f"BG_loss_{combined_loss[0]}_____________NC_loss_{combined_loss[1]}___________ED_loss_{combined_loss[2]}_____________ET_loss_{combined_loss[3]}")
                print(f"BG_loss_{combined_loss[0]}__________ET_loss_{combined_loss[1]}")

                loss_BG = combined_loss[0]
                
                loss_EN = combined_loss[1]

                
                loss = loss_EN+loss_BG
    
                
                batch_images = batch['mask'].shape[0]
               

                for idx, (key, value) in enumerate(class_losses_sum_overall_wo.items()):
                    class_losses_sum_overall_wo[key]+=((combined_loss[idx].item())*batch_images)

                # for idx, (key, value) in enumerate(class_losses_sum_overall.items()):
                #     class_losses_sum_overall[key]+=((combined_loss_bts[idx].item())*batch_images)
                loss = loss*batch_images
                
                if isinstance(hd_95, torch.Tensor):
                    hd95_np = hd_95.detach().cpu().numpy()  # Ensure it's detached from the graph
                else:
                    hd95_np = np.array(hd_95)  # Convert other types
                
                print(f"hd95_np type: {type(hd95_np)}, value: {hd95_np}")
                
                # Ensure it's a scalar value
                if isinstance(hd95_np, np.ndarray) and hd95_np.size == 1:
                    hd95_np = hd95_np.item()  # Convert single-element array to scalar
                
                # Check for NaN or Inf before appending
                if np.isfinite(hd95_np):
                    hd_95npes.append(hd95_np)
                else:
                    print("Skipping NaN or Inf HD95 value")
               
                latent_losses += loss.item()
            # indices_losses += indices_loss.item()
    for key, value in class_losses_sum_overall_wo.items():
        class_losses_sum_overall_wo[key] = value / val_dataset_len
    
    for key, value in class_losses_sum_overall.items():
        class_losses_sum_overall[key] = value / val_dataset_len
    latent_losses = latent_losses / val_dataset_len  
    # hd_95es = class_losses_sum_overall_wo['ET']
    hd_95es =torch.mean(torch.tensor(hd_95npes, dtype=torch.float32))
    print("val_steps", hd_95es)
    print("sum of hd95_es", sum(hd_95npes))
    print("len of hd95_es", len(hd_95npes))
    # Return the average loss over the validation dataset
    return latent_losses, class_losses_sum_overall, class_losses_sum_overall_wo, hd_95es


def test_cond(model_WT, autoencoder_WT, model_TC, autoencoder_TC, model_ET, autoencoder_ET, dataloader, val_dataset_len, device):
    """
    Validate the VAE model on the validation dataset.
    """
    print("Validation in Progress")
    # model = model.to(device)
    model_WT.eval()  # Set the model to evaluation mode
    model_TC.eval()
    model_ET.eval()
    autoencoder_WT.eval()
    autoencoder_TC.eval()
    autoencoder_ET.eval()
    val_loss = 0  # Initialize total loss accumulator
    q_losses = 0
    mse_losses = 0
    quantization_losses = 0
    latent_losses = 0
    indices_losses = 0
    class_losses_sum_overall_wo = {'WT': 0, 'TC': 0, 'ET': 0}
    class_losses_sum_overall = {'BG': 0, 'TC': 0, 'WT': 0, 'ET': 0}

    hd_95_npes = {"WT_HD_95": 0, "TC_HD_95": 0, "ET_HD_95": 0}
    count = 0
    # class_losses_sum_overall = {'BG': 0, 'NC': 0}
    with torch.no_grad():  # Disable gradient computation for validation
        
        for batch in tqdm(dataloader, desc="Inference"):
            
                       
            images={}
            for key in ["t1n", "t2w", "t1c", "t2f"]:
                if key in batch:
                    images[key] = (batch[key])
                   # print(f"image shape with modality {key} is", batch[key].shape)
                else:
                    raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
        
           
            images = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1)
           
            t2f = batch['t2f_normalized']  # Reverse normalization for t2f

            if 'mask' in batch:
                mask = batch['mask']
                # print("image shape with seg_mask is", mask.shape)
            else:
                raise KeyError("Key 'segmentation' not found in batch_data") 

            mask = mask.to(device)
            mask_up_WT = mask[:,1:2,:,:,:]
            mask_up_TC = mask[:,2:3,:,:,:]
            mask_up_ET = mask[:,3:,:,:,:]
            
            images = images.to(device)
            # mask_up = mask_up*images
            autoencoder_WT_autoencoder_latent=autoencoder_WT.encoder(mask_up_WT) 
            autoencoder_WT_autoencoder_latent = autoencoder_WT.bottleneck(autoencoder_WT_autoencoder_latent)
            autoencoder_WT_autoencoder_latent=autoencoder_WT.conv1(autoencoder_WT_autoencoder_latent) 
            autoencoder_WT_autoencoder_latent=autoencoder_WT.conv2(autoencoder_WT_autoencoder_latent) 
            autoencoder_latent_indices_WT=autoencoder_WT.quantizer0.quantize(autoencoder_WT_autoencoder_latent)
            autoencoder_latent_indices_embeddingsss_WT = autoencoder_WT.quantizer0.embed(autoencoder_latent_indices_WT)
            autoencoder_latent_indices_WT = autoencoder_latent_indices_embeddingsss_WT.long()


            autoencoder_TC_autoencoder_latent=autoencoder_TC.encoder(mask_up_TC) 
            autoencoder_TC_autoencoder_latent = autoencoder_TC.bottleneck(autoencoder_TC_autoencoder_latent)
            autoencoder_TC_autoencoder_latent=autoencoder_TC.conv1(autoencoder_TC_autoencoder_latent) 
            autoencoder_TC_autoencoder_latent=autoencoder_TC.conv2(autoencoder_TC_autoencoder_latent) 
            autoencoder_latent_indices_TC=autoencoder_TC.quantizer0.quantize(autoencoder_TC_autoencoder_latent)
            autoencoder_latent_indices_embeddingsss_TC = autoencoder_TC.quantizer0.embed(autoencoder_latent_indices_TC)
            autoencoder_latent_indices_TC = autoencoder_latent_indices_embeddingsss_TC.long()



            autoencoder_ET_autoencoder_latent=autoencoder_ET.encoder(mask_up_ET) 
            autoencoder_ET_autoencoder_latent = autoencoder_ET.bottleneck(autoencoder_ET_autoencoder_latent)
            autoencoder_ET_autoencoder_latent=autoencoder_ET.conv1(autoencoder_ET_autoencoder_latent) 
            autoencoder_ET_autoencoder_latent=autoencoder_ET.conv2(autoencoder_ET_autoencoder_latent) 
            autoencoder_latent_indices_ET=autoencoder_ET.quantizer0.quantize(autoencoder_ET_autoencoder_latent)
            autoencoder_latent_indices_embeddingsss_ET = autoencoder_ET.quantizer0.embed(autoencoder_latent_indices_ET)
            autoencoder_latent_indices_ET = autoencoder_latent_indices_embeddingsss_ET.long()
            
            with autocast(device_type='cuda', enabled=False):                 
                
                x_bot_WT, x_bottt, quantized, quantized_loss =  model_WT(images)
                x_bot_TC, x_bottt, quantized, quantized_loss =  model_TC(images)
                x_bot_ET, x_bottt, quantized, quantized_loss =  model_ET(images)
                
                x_bot_WT = torch.argmax(x_bot_WT, dim=1)
                
                embeddingsss_WT = autoencoder_WT.quantizer0.embed(x_bot_WT)
               
                reconstruction_WT = autoencoder_WT.conv3(embeddingsss_WT)
                reconstruction_WT = autoencoder_WT.conv4(reconstruction_WT)
                reconstruction_WT = autoencoder_WT.decoder(reconstruction_WT)
                reconstruction_WT = autoencoder_WT.segmentation(reconstruction_WT)
                reconstruction_WT_mask = torch.argmax(reconstruction_WT, dim=1)
                reconstruction_WT = reconstruction_WT[:,1:,:,:,:]


                x_bot_TC = torch.argmax(x_bot_TC, dim=1)
                
                embeddingsss_TC = autoencoder_TC.quantizer0.embed(x_bot_TC)
               
                reconstruction_TC = autoencoder_TC.conv3(embeddingsss_TC)
                reconstruction_TC = autoencoder_TC.conv4(reconstruction_TC)
                reconstruction_TC = autoencoder_TC.decoder(reconstruction_TC)
                reconstruction_TC = autoencoder_TC.segmentation(reconstruction_TC)
                reconstruction_TC_mask = torch.argmax(reconstruction_TC, dim=1)
                reconstruction_TC = reconstruction_TC[:,1:,:,:,:]


                x_bot_ET = torch.argmax(x_bot_ET, dim=1)
                
                embeddingsss_ET = autoencoder_ET.quantizer0.embed(x_bot_ET)
               
                reconstruction_ET = autoencoder_ET.conv3(embeddingsss_ET)
                reconstruction_ET = autoencoder_ET.conv4(reconstruction_ET)
                reconstruction_ET = autoencoder_ET.decoder(reconstruction_ET)
                reconstruction_ET = autoencoder_ET.segmentation(reconstruction_ET)
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
               

                for idx, (key, value) in enumerate(class_losses_sum_overall_wo.items()):
                    class_losses_sum_overall_wo[key]+=(1-(combined_loss[idx].item())*batch_images)

                # for idx, (key, value) in enumerate(class_losses_sum_overall.items()):
                #     class_losses_sum_overall[key]+=((combined_loss_bts[idx].item())*batch_images)
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
                
                # Check for NaN or Inf before appending
                if np.all(np.isfinite(np.array(list(hd_95.values())))):
                    hd_95_npes = {key: hd_95_npes[key] + hd_95[key] for key in hd_95_npes}
                    
                # if np.isfinite(hd95_np):
                #     hd_95_npes += hd95_np
                    print("HD95 per class:", hd_95_npes)
                    count+=1
                else:
                    print("Skipping NaN or Inf HD95 value")

    
            
            # val_loss += loss_val.item()  # Accumulate the loss value
            # q_losses += q_loss.item()
            # mse_losses += mse_loss_val.item()
                latent_losses += loss.item()
            # indices_losses += indices_loss.item()
    for key, value in class_losses_sum_overall_wo.items():
        class_losses_sum_overall_wo[key] = value / val_dataset_len
    
    for key, value in class_losses_sum_overall.items():
        class_losses_sum_overall[key] = value / val_dataset_len

    for key, value in hd_95_npes.items():
        hd_95_npes[key] = value / count

    print("class_losses_sum_overall_wo", class_losses_sum_overall_wo)
    print("class_losses_sum_overall", class_losses_sum_overall)

    print("hd_95_npes", hd_95_npes)

    print("mask shape is", mask.shape)
    print("reconstruction shape is", reconstruction.shape)
    # latent_losses = latent_losses / val_dataset_len  
    # Return the average loss over the validation dataset
    return mask, reconstruction, batch['t2f']
            # mask = torch.argmax(mask, dim=1)
            # reconstruction = torch.argmax(reconstruction, dim=1) 
            # if combined_loss[1].item()>=0.2:
            # yield final_GT_segmentation_mask, final_segmentation_mask, t2f, class_losses_sum_overall_wo, hd_95

