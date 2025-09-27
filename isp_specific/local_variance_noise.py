import torch
import torch.nn.functional as F
import cv2
import numpy as np


def local_variance_noise_torch(
        img: torch.Tensor,
        window_size: int = 7,
        use_texture_mask: bool = False
) -> torch.Tensor:
    """
    Estimate luminance noise level using local variance.

    This function computes the mean local variance in the luminance channel
    as a proxy for noise energy. Note that this metric includes both noise
    and texture, so it may overestimate noise in textured regions.

    Args:
        img (torch.Tensor): Input RGB image tensor of shape [B, 3, H, W].
                            Expected to be normalized to [0, 1] range.
        window_size (int): Size of the local window for variance estimation.
                           Must be odd. Default: 7.
        use_texture_mask (bool): If True, apply a simple texture mask to
                                reduce contribution from high-texture regions.
                                (Advanced feature, not implemented here)

    Returns:
        torch.Tensor: Scalar tensor representing average local variance.
                      Higher value = more noise/texture.
                      Value is in squared intensity units.

    Note:
        - This metric is commonly used as a fast noise proxy in ISP pipelines.
        - For pure noise estimation, consider:
          * Using smooth regions only (e.g., via edge detection)
          * Multi-scale analysis
        - Local variance = E[X²] - (E[X])²
    """
    assert img.dim() == 4 and img.shape[1] == 3, "Input must be [B, 3, H, W]"
    assert window_size % 2 == 1, "window_size must be odd"

    # --- Convert to luminance (BT.601) ---
    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]  # [B, H, W]
    gray = gray.unsqueeze(1)  # [B, 1, H, W]

    # --- Compute local variance ---
    # Use reflect padding to avoid border artifacts
    pad = window_size // 2
    gray_pad = F.pad(gray, (pad, pad, pad, pad), mode='reflect')

    # Uniform averaging kernel
    kernel = torch.ones(
        1, 1, window_size, window_size,
        device=img.device,
        dtype=img.dtype
    ) / (window_size ** 2)

    # Local mean and mean of squares
    mean = F.conv2d(gray_pad, kernel)
    mean_sq = F.conv2d(gray_pad ** 2, kernel)

    # Local variance: Var = E[X²] - (E[X])²
    var = mean_sq - mean ** 2

    # Clamp negative variances (caused by floating-point precision)
    var = torch.clamp(var, min=0.0)

    # Average over all pixels and batch
    noise_level = var.mean()
    return noise_level


if __name__ == '__main__':
    img = torch.rand((1, 3, 224, 224)).cuda()
    print(local_variance_noise_torch(img))
