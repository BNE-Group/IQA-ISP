import torch
import torch.nn.functional as F
import cv2
import numpy as np
import piq.ssim


def ssim_torch(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    return piq.ssim(img1, img2)


if __name__ == '__main__':
    img1 = torch.rand((1, 3, 224, 224)).cuda()
    img2 = torch.rand((1, 3, 224, 224)).cuda()
    print(ssim_torch(img1, img2))
