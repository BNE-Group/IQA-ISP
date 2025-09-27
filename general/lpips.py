import torch
import torch.nn.functional as F
import cv2
import numpy as np
from piq import LPIPS

net = LPIPS()

def lpips_torch(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    return net.to(img1.device)(img1, img2)


if __name__ == ('__'
                'main__'):
    img1 = torch.rand((1, 3, 224, 224)).cuda()
    img2 = torch.rand((1, 3, 224, 224)).cuda()
    print(lpips_torch(img1, img2))
