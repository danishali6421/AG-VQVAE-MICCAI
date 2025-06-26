import monai
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.config import DtypeLike, KeysCollection
from monai.transforms import Transform, Compose, MapTransform
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    RandSpatialCropd,
    SpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAffined,
    RandRotated,
    ToTensord,
    Resized,
    ConcatItemsd,
    CenterSpatialCropd,
)
from config.configp import get_args
import numpy as np
import torch



# from __future__ import annotations

import logging
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import partial
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from collections.abc import Callable, Hashable, Mapping
from monai.config import DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import is_no_channel, no_collation
from monai.networks.layers.simplelayers import (
    ApplyFilter,
    EllipticalFilter,
    GaussianFilter,
    LaplaceFilter,
    MeanFilter,
    SavitzkyGolayFilter,
    SharpenFilter,
    median_filter,
)
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.traits import MultiSampleTrait
from monai.transforms.transform import Randomizable, RandomizableTrait, RandomizableTransform, Transform
from monai.transforms.utils import (
    extreme_points_to_image,
    get_extreme_points,
    map_binary_to_indices,
    map_classes_to_indices,
)
from monai.transforms.utils_pytorch_numpy_unification import concatenate, in1d, moveaxis, unravel_indices
from monai.utils import (
    MetaKeys,
    TraceKeys,
    convert_data_type,
    convert_to_cupy,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    look_up_option,
    min_version,
    optional_import,
)
from monai.utils.enums import TransformBackends
from monai.utils.misc import is_module_ver_at_least
from monai.utils.type_conversion import convert_to_dst_type, get_equivalent_dtype

import cv2
import numpy as np
from monai.transforms import Transform, MapTransform
from typing import Hashable, Mapping
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, binary_closing
# from skimage.morphology import binary_dilation, binary_erosion, binary_opening, binary_closing
import random
from skimage.measure import label, regionprops



class ConvertToMultiChannelBasedOnBratsClasses_bbox(Transform):
    """
    Convert labels to multi channels based on BRATS classes:
    - Label 1 is the necrotic and non-enhancing tumor core
    - Label 2 is the peritumoral edema
    - Label 4 is the GD-enhancing tumor
    This implementation crops the region of interest dynamically 
    to include additional background around the bounding box to 
    meet the target size.
    """

    def __init__(self, target_size=(120, 120, 155)):
        """
        Args:
            target_size (tuple): Desired crop size (height, width, depth).
        """
        self.target_size = target_size

    def crop_to_target_size(self, img, bbox):
        """
        Crop the image to include the bounding box and extend the region to match the target size
        by including additional background.

        Args:
            img (numpy.ndarray): The input 3D image.
            bbox (tuple): The bounding box (x_min, x_max, y_min, y_max, z_min, z_max).
            target_size (tuple): The desired crop size (height, width, depth).

        Returns:
            numpy.ndarray: The cropped image.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        crop_h, crop_w, crop_d = self.target_size

        # Calculate bounding box dimensions
        bbox_h = x_max - x_min
        bbox_w = y_max - y_min
        bbox_d = z_max - z_min

        # Calculate required extensions
        extra_h = max(0, crop_h - bbox_h)
        extra_w = max(0, crop_w - bbox_w)
        extra_d = max(0, crop_d - bbox_d)

        # Extend bounding box symmetrically
        start_x = max(0, x_min - extra_h // 2)
        end_x = min(img.shape[0], x_max + extra_h // 2)

        start_y = max(0, y_min - extra_w // 2)
        end_y = min(img.shape[1], y_max + extra_w // 2)

        start_z = max(0, z_min - extra_d // 2)
        end_z = min(img.shape[2], z_max + extra_d // 2)

        # Adjust for smaller dimensions
        if end_x - start_x < crop_h:
            start_x = max(0, end_x - crop_h)
        if end_y - start_y < crop_w:
            start_y = max(0, end_y - crop_w)
        if end_z - start_z < crop_d:
            start_z = max(0, end_z - crop_d)

        # Final crop
        cropped_img = img[start_x:start_x + crop_h, start_y:start_y + crop_w, start_z:start_z + crop_d]

        crop_coords = [start_x, crop_h, start_y, crop_w, start_z, crop_d]

        return cropped_img, crop_coords

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # If img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        combined_tumor = (img == 1) | (img == 2) | (img == 4)
        labeled_mask = label(combined_tumor)

        coordinates = np.argwhere(labeled_mask > 0)


        min_x, min_y, min_z = coordinates.min(axis=0)
        max_x, max_y, max_z = coordinates.max(axis=0)
        bbox = (min_x, max_x, min_y, max_y, min_z, max_z)

        cropped_img, crop_coords = self.crop_to_target_size(img, bbox)

        return cropped_img, crop_coords



class ConvertToMultiChannelBasedOnBratsClassesd_bbox(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = ConvertToMultiChannelBasedOnBratsClasses_bbox.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBratsClasses_bbox()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key == 'mask':
                cropped_img, crop_coords = self.converter(d[key])
                d[key] = cropped_img
        return d




class ConvertToMultiChannelBasedOnBratsClasses(Transform):
    """
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        uni_labels = np.array([0, 1, 2, 3])        
        if np.array_equal(np.unique(img), uni_labels):
            # print("All labels are there")
            missing_labels = None
        else:
            missing_labels = np.setdiff1d(uni_labels, np.unique(img))
            missing_labels=missing_labels[0]
        result = [(img == 0), (img == 1)|(img == 3)|(img==2), (img == 1)|(img == 3), (img == 3)]
        return torch.stack(result, dim=0).float() if isinstance(img, torch.Tensor) else np.stack(result, axis=0).astype(float)

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = ConvertToMultiChannelBasedOnBratsClasses.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBratsClasses()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d



class RandomMorphologyTransform(Transform):
    def __init__(self, operations=None, min_kernel_size=1, max_kernel_size=5, min_iterations=1, max_iterations=3):
        """
        Perform random morphological operations (dilation, erosion, opening, or closing) on 3D volumes.

        Args:
            operations (list of str): The types of morphological operations to randomly select from.
            min_kernel_size (int): Minimum size of the structuring element (cube in 3D).
            max_kernel_size (int): Maximum size of the structuring element (cube in 3D).
            min_iterations (int): Minimum number of times the operation is applied.
            max_iterations (int): Maximum number of times the operation is applied.
        """
        self.operations = operations if operations else ['dilation', 'erosion']
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations

    def apply_morphology(self, volumes: np.ndarray) -> np.ndarray:
        # Randomly select operation, kernel size, and iterations
        operation = random.choice(self.operations)
        kernel_size = random.randint(self.min_kernel_size, self.max_kernel_size)
        iterations = random.randint(self.min_iterations, self.max_iterations)

        # Define a 3D structuring element (cube)
        struct_elem = np.ones((kernel_size, kernel_size, kernel_size), dtype=np.uint8)

        num_indices_to_apply = random.choice([1])

        # print("num_indices_to_apply", num_indices_to_apply)
        
        # Randomly select the specific indices to apply the operation to
        indices_to_apply = random.sample([1], num_indices_to_apply)

        # print("indices_to_apply", indices_to_apply)

        # Apply the selected morphological operation to the randomly chosen indices
        if operation == 'closing':
            # print(f"Applying Closing on indices {indices_to_apply}")
            return np.array([
                binary_closing(volume, structure=struct_elem, iterations=iterations).astype(volume.dtype)
                if idx in indices_to_apply else volume
                for idx, volume in enumerate(volumes)
            ])
        elif operation == 'erosion':
            print(f"Applying Erosion on indices {indices_to_apply}")
            return np.array([
                binary_erosion(volume, structure=struct_elem, iterations=iterations).astype(volume.dtype)
                if idx in indices_to_apply else volume
                for idx, volume in enumerate(volumes)
            ])
        elif operation == 'opening':
            # print(f"Applying Opening on indices {indices_to_apply}")
            return np.array([
                binary_opening(volume, structure=struct_elem, iterations=iterations).astype(volume.dtype)
                if idx in indices_to_apply else volume
                for idx, volume in enumerate(volumes)
            ])
        elif operation == 'dilation':
            print(f"Applying Dilation on indices {indices_to_apply}")
            return np.array([
                binary_dilation(volume, structure=struct_elem, iterations=iterations).astype(volume.dtype)
                if idx in indices_to_apply else volume
                for idx, volume in enumerate(volumes)
            ])
        return volumes

    def __call__(self, volumes: np.ndarray) -> np.ndarray:
        if volumes.ndim == 5:
            # Apply transformation to each volume in parallel
            return self.apply_morphology(volumes)
        elif volumes.ndim == 4:
            # Apply transformation to each volume in parallel
            return self.apply_morphology(volumes)
        else:
            raise ValueError(f"Expected a 4D or 5D tensor, got {volumes.ndim}D.")

class RandomMorphologyTransformd(MapTransform):
    """
    Dictionary-based wrapper for RandomMorphologyTransform to work with MONAI pipelines.
    """
    def __init__(self, keys: Hashable, operations=None, min_kernel_size=1, max_kernel_size=5, 
                 min_iterations=1, max_iterations=3, prob=0.5, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.transform = RandomMorphologyTransform(operations, min_kernel_size, max_kernel_size, min_iterations, max_iterations)
        self.prob = prob
        self.count=0
        

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            volume = d[key]
            d['wo_morphological'] = volume
            # Convert torch tensor to numpy array if necessary
            if isinstance(volume, torch.Tensor):
                volume = volume.cpu().numpy()

            # Apply the random morphological transformation on the volume
            if random.random() < 0.5:
                # print("VOLUME SHAPE IS", volume.shape)
                # for idx, volum in enumerate(volume):
                #     print(f"idx is____{idx}_________volume shape is__{volum.shape}")
                volume = self.transform(volume)                
            # Convert back to torch tensor and ensure type consistency
            d[key] = torch.from_numpy(volume).float()
            
            # print("d[key]", d[key].shape)
        self.count += 1
        # print("self.count", self.count)
        return d


class PrintDtypeTransform(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            print(f"Key: {key}, dtype: {data[key].dtype}, shape: {data[key].shape}")
        return data

class SqueezeDimsTransform(Transform):
    def __init__(self, keys, dims):
        self.keys = keys
        self.dims = dims

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                # Remove specified dimensions
                # print("Data old shape", data[key].shape)
                data[key] = np.squeeze(data[key], axis=self.dims)
                # print("Data new shape", data[key].shape)
        return data

class TrackingTransform(Transform):
    def __init__(self, transform: Transform, name: str):
        self.transform = transform
        self.name = name
        self.count = 0  # Initialize a count for tracking how many times this transformation is applied

    def __call__(self, data):
        result = self.transform(data)
        self.count += 1  # Increment count each time this transformation is applied
        return result

    def __repr__(self):
        return f"{self.name}({self.transform}), applied {self.count} times"





class StoreMeanStdTransform:
    def __init__(self, keys, device='cpu'):
        self.keys = keys
        self.device = device
        self.normalize_transform = NormalizeIntensityd(keys=self.keys)
    def __call__(self, data):
        # First, store mean and std from the original data (with non-zero values)
        for key in self.keys:
            if key in data:
                normalized_image = self.normalize_transform(data)[key]  # Normalize the image for this key
                
                # Store the normalized image in a new key
                data[f"{key}_normalized"] = normalized_image
            else:
                raise KeyError(f"Key {key} not found in data")

        # No need to apply normalization here since it's already handled elsewhere
        return data




def get_train_transforms(crop_size):
    print("crop_size is", crop_size)
    return monai.transforms.Compose([
        LoadImaged(keys=["t1n", "t2w", "t1c", "t2f", 'mask']),  # Assuming images and masks are loaded from file paths
        # PrintDtypeTransform(keys=["t1", "t2", "t1ce", "flair", 'mask']),
        EnsureChannelFirstd(keys=["t1n", "t2w", "t1c", "t2f", 'mask']),
        NormalizeIntensityd(keys=["t1n", "t2w", "t1c", "t2f"], nonzero=True),
        # ConvertToMultiChannelBasedOnBratsClassesd_bbox(keys=['mask']),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
        # Resized(keys=["t1n", "t2w", "t1c", "t2f"], spatial_size=(120, 120, 120), mode="nearest"),
        RandRotated(keys=["t1n", "t2w", "t1c", "t2f", 'mask'], prob=0.3, range_x=(-15, 15), range_y=(-15, 15), range_z=(-15, 15), mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")),
        RandFlipd(keys=["t1n", "t2w", "t1c", "t2f", 'mask'], prob=0.3, spatial_axis=0),
        # PrintDtypeTransform(keys=["t1", "t2", "t1ce", "flair", 'mask']),
        RandFlipd(keys=["t1n", "t2w", "t1c", "t2f", 'mask'], prob=0.3, spatial_axis=1),
        # PrintDtypeTransform(keys=["t1", "t2", "t1ce", "flair", 'mask']),
        RandFlipd(keys=["t1n", "t2w", "t1c", "t2f", 'mask'], prob=0.3, spatial_axis=2),
        # PrintDtypeTransform(keys=["t1", "t2", "t1ce", "flair", 'mask']),
        # PrintDtypeTransform(keys=["t1", "t2", "t1ce", "flair", 'mask']),
        RandScaleIntensityd(keys=["t1n", "t2w", "t1c", "t2f"], factors=0.1, prob=0.1),
        # PrintDtypeTransform(keys=["t1", "t2", "t1ce", "flair", 'mask']),
        RandShiftIntensityd(keys=["t1n", "t2w", "t1c", "t2f"], offsets=0.1, prob=0.1),
        # PrintDtypeTransform(keys=["t1", "t2", "t1ce", "flair", 'mask']),
        SqueezeDimsTransform(keys=["t1n", "t2w", "t1c", "t2f"], dims=0),
        # PrintDtypeTransform(keys=["t1", "t2", "t1ce", "flair", 'mask']),
        # RandomMorphologyTransformd(keys=['mask'], min_kernel_size=1, max_kernel_size=3, min_iterations=1, max_iterations=3, prob=0.2),
        ToTensord(keys=["t1n", "t2w", "t1c", "t2f", 'mask']),
        # PrintDtypeTransform(keys=["t1", "t2", "t1ce", "flair", 'mask'])
    ])


def get_val_transforms(crop_size):
    return monai.transforms.Compose([
        LoadImaged(keys=["t1n", "t2w", "t1c", "t2f", 'mask']),  # Assuming images and masks are loaded from file paths
        EnsureChannelFirstd(keys=["t1n", "t2w", "t1c", "t2f", 'mask']),
        StoreMeanStdTransform(keys=["t1n", "t2w", "t1c", "t2f"]),
        NormalizeIntensityd(keys=["t1n", "t2w", "t1c", "t2f"], nonzero=True),
        
        ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
        
        SqueezeDimsTransform(keys=["t1n", "t2w", "t1c", "t2f"], dims=0),
        
        ToTensord(keys=["t1n", "t2w", "t1c", "t2f", 'mask']),
    ])