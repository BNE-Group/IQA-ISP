import torch
import torch.nn.functional as F
import cv2
import numpy as np
import piq.psnr


def psnr_torch(img1: torch.Tensor, img2:torch.Tensor) -> torch.Tensor:
    return piq.psnr(img1, img2)


if __name__ == '__main__':
    img1 = torch.rand((1, 3, 224, 224)).cuda()
    img2 = torch.rand((1, 3, 224, 224)).cuda()
    print(psnr_torch(img1, img2))
