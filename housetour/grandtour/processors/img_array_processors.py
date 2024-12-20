"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from torchvision import transforms
from PIL import Image
from omegaconf import OmegaConf
from timm.layers import to_2tuple

from grandtour.processors import functional_video as F


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__

class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__

class NormalizeVideo:
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        return F.normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"

class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
        """
        return F.to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__

class ImgArrayProcessor():
    def __init__(
        self,
        image_size=None,
        mean=None,
        std=None,
        n_frms=None,
        min_scale=None,
        max_scale=None,
    ):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = NormalizeVideo(mean, std)
        self.n_frms = n_frms
        self.image_size = image_size if type(image_size) == tuple else to_2tuple(image_size)
        self.transform = transforms.Compose(
            [
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                ToTensorVideo(),  # C, T, H, W
                self.normalize,  # C, T, H, W
            ]
        )
    
    def __call__(self, img_array_path):
        """
        Args:
            img_array_path (List[str]): Paths for the image array
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        img_array = [transforms.PILToTensor()(Image.open(img_path)) for img_path in img_array_path]
        img_array = torch.stack(img_array)
        img_array = transforms.Resize((self.image_size[0], self.image_size[1]))(img_array)
        img_array = img_array.permute(1, 0, 2, 3)
        return self.transform(img_array)
    
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        n_frms = cfg.get("n_frms", 8)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frms=n_frms,
        )

