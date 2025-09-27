import torch
import torch.nn.functional as F
import cv2
import numpy as np


def luma_noise_level_torch(
        img: torch.Tensor,
        window_size: int = 7
) -> torch.Tensor:
    """
    Estimate luminance noise level using local variance.

    This function computes the mean local variance in the luminance channel
    as a proxy for brightness noise energy. Note that this metric includes
    both noise and texture, so it may overestimate noise in textured regions.

    Args:
        img (torch.Tensor): Input RGB image tensor of shape [B, 3, H, W].
                            Values should be normalized to [0, 1] range.
        window_size (int): Size of the local window for variance estimation.
                           Recommended to be an odd integer (e.g., 5, 7, 9).
                           Default: 7.

    Returns:
        torch.Tensor: Scalar tensor representing average luminance noise level.
                      Higher value = more noise (or texture).
                      Units: squared intensity (e.g., if input in [0,1], output in [0,1]).

    Note:
        - This is a fast, no-reference noise estimator commonly used in ISP pipelines.
        - For pure noise estimation, consider restricting to smooth regions (e.g., via edge detection).
        - Typical values:
            * Clean image: < 0.001
            * Moderate noise: 0.001–0.01
            * Heavy noise: > 0.01
    """
    # Input validation
    if img.dim() != 4 or img.shape[1] != 3:
        raise ValueError("Input must be a 4D tensor of shape [B, 3, H, W]")

    # Convert to luminance (BT.601 standard)
    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]  # [B, H, W]
    gray = gray.unsqueeze(1)  # [B, 1, H, W]

    # Padding to maintain spatial dimensions
    pad = window_size // 2
    gray_padded = F.pad(gray, (pad, pad, pad, pad), mode='reflect')
    gray_sq_padded = F.pad(gray ** 2, (pad, pad, pad, pad), mode='reflect')

    # Compute local mean and mean of squares using average pooling
    mean = F.avg_pool2d(gray_padded, kernel_size=window_size, stride=1)
    mean_sq = F.avg_pool2d(gray_sq_padded, kernel_size=window_size, stride=1)

    # Local variance: Var = E[X^2] - (E[X])^2
    var = mean_sq - mean ** 2

    # Clamp negative variances caused by floating-point precision errors
    var = torch.clamp(var, min=0.0)

    # Return mean variance across all pixels and batch
    return var.mean()


if __name__ == '__main__':
    img = torch.rand((1, 3, 224, 224)).cuda()
    print(luma_noise_level_torch(img))
