import torch
import torch.nn.functional as F
import cv2
import numpy as np


def tenengrad_sharpness_torch(img: torch.Tensor) -> torch.Tensor:
    """
    Compute image sharpness using the Tenengrad method.

    Tenengrad sharpness is defined as the mean squared gradient magnitude,
    which correlates well with perceived image sharpness.

    Args:
        img (torch.Tensor): Input RGB image tensor of shape [B, 3, H, W].
                            Values should be normalized to [0, 1] range.

    Returns:
        torch.Tensor: Scalar tensor representing average sharpness.
                      Higher value = sharper image.
                      Units: squared intensity per pixel.

    Note:
        - This metric is sensitive to noise; consider combining with noise metrics.
        - Typical values:
            * Blurry image: < 0.01
            * Normal sharpness: 0.01–0.1
            * Over-sharpened: > 0.1
        - Based on: P. K. S. Tamura et al., "Tenengrad and its variants"
    """
    # Input validation
    if img.dim() != 4 or img.shape[1] != 3:
        raise ValueError("Input must be a 4D tensor of shape [B, 3, H, W]")

    device, dtype = img.device, img.dtype

    # --- Convert to luminance (BT.601 standard) ---
    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]  # [B, H, W]
    gray = gray.unsqueeze(1)  # [B, 1, H, W]

    # --- Define Sobel operators ---
    # Sobel kernels for gradient computation
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=dtype, device=device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=dtype, device=device).view(1, 1, 3, 3)

    # --- Compute gradients with padding to maintain spatial size ---
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)

    # --- Compute Tenengrad sharpness: mean of squared gradient magnitude ---
    sharpness = (gx ** 2 + gy ** 2).mean()
    return sharpness

if __name__ == '__main__':
    img = torch.rand((1, 3, 224, 224)).cuda()
    print(tenengrad_sharpness_torch(img))
