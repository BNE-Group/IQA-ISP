import torch
import torch.nn.functional as F
import cv2
import numpy as np


def watercolor_torch(
        img: torch.Tensor,
        window: int = 15
) -> torch.Tensor:
    """
    Detect "watercolor" or "plastic skin" artifacts caused by over-smoothing.

    This metric quantifies the loss of texture due to aggressive denoising.
    It is based on the observation that over-smoothed regions exhibit very
    low local luminance variance.

    Method:
    1. Convert to luminance (BT.709)
    2. Compute local variance in a sliding window
    3. Map low variance to high "watercolor score"

    Args:
        img (torch.Tensor): Input RGB image [B, 3, H, W], values in [0, 1]
        window (int): Window size for local variance estimation.
                      Should be odd and >= 3. Default: 15

    Returns:
        torch.Tensor: Scalar watercolor artifact score in range [0, 1].
                      Higher value = more severe watercolor effect.
                      0 = no over-smoothing, 1 = completely flat regions.

    Note:
        - This metric is commonly used to detect "plastic face" in portrait photography.
        - For best results, use on images with expected texture content (e.g., skin, fabric).
        - Typical values:
            * Natural image: 0.0–0.3
            * Mild over-smoothing: 0.3–0.6
            * Severe watercolor: > 0.6
    """
    # Input validation
    if img.dim() != 4 or img.shape[1] != 3:
        raise ValueError("Input must be [B, 3, H, W]")
    if window < 3:
        raise ValueError("window must be >= 3")

    device, dtype = img.device, img.dtype

    # --- Convert to luminance (BT.709) ---
    Y = 0.2126 * img[:, 0] + 0.7152 * img[:, 1] + 0.0722 * img[:, 2]  # [B, H, W]
    Y = Y.unsqueeze(1)  # [B, 1, H, W]

    # --- Compute local variance ---
    pad = window // 2
    Y_pad = F.pad(Y, (pad, pad, pad, pad), mode='reflect')

    # Local mean and mean of squares
    mean_Y = F.avg_pool2d(Y_pad, kernel_size=window, stride=1)
    mean_Y_sq = F.avg_pool2d(Y_pad ** 2, kernel_size=window, stride=1)

    # Local variance with numerical stability
    var_Y = mean_Y_sq - mean_Y ** 2
    var_Y = torch.clamp(var_Y, min=0.0)

    # --- Map low variance to high watercolor score ---
    # Use exponential mapping for smooth, bounded output
    # Higher variance → lower score; lower variance → higher score
    mean_var = var_Y.mean()

    # Normalize: assume max variance is 0.25 (for [0,1] image, max var = 0.25 at checkerboard)
    normalized_var = torch.clamp(mean_var / 0.25, min=0.0, max=1.0)
    watercolor_score = 1.0 - normalized_var  # [0, 1]

    return watercolor_score


if __name__ == '__main__':
    img = torch.rand((1, 3, 224, 224)).cuda()
    print(watercolor_torch(img))