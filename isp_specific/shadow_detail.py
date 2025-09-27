import torch
import torch.nn.functional as F
import cv2
import numpy as np


def shadow_detail_torch(
        img: torch.Tensor,
        shadow_threshold: float = 0.2
) -> torch.Tensor:
    """
    Quantify shadow detail preservation in an image.

    This metric evaluates how well the ISP preserves texture and contrast
    in dark regions, which is critical for low-light image quality.

    Method:
    1. Identify shadow regions (luminance < shadow_threshold)
    2. Compute local contrast as |Y - blurred_Y|
    3. Average local contrast over shadow regions

    Args:
        img (torch.Tensor): Input RGB image [B, 3, H, W], values in [0, 1]
        shadow_threshold (float): Luminance threshold to define shadow regions.
                                 Typical values: 0.1 (very dark) to 0.3 (mid-shadow).
                                 Must be in [0, 1]. Default: 0.2

    Returns:
        torch.Tensor: Scalar score representing average shadow detail.
                      Higher value = more detail preserved in shadows.
                      Value is in intensity units (same as input luminance).

    Note:
        - This metric is sensitive to noise; consider combining with noise metrics.
        - For best results, use on images with actual shadow content.
        - Typical values:
            * Dead black (no detail): ~0.0
            * Moderate detail: 0.02–0.05
            * Rich shadow detail: > 0.05
    """
    # Input validation
    if img.dim() != 4 or img.shape[1] != 3:
        raise ValueError("Input must be [B, 3, H, W]")
    if not (0.0 <= shadow_threshold <= 1.0):
        raise ValueError("shadow_threshold must be in [0, 1]")

    device, dtype = img.device, img.dtype

    # --- Convert to luminance (BT.709) ---
    Y = 0.2126 * img[:, 0] + 0.7152 * img[:, 1] + 0.0722 * img[:, 2]  # [B, H, W]

    # --- Identify shadow regions ---
    shadow_mask = Y < shadow_threshold  # [B, H, W]

    # --- Compute local contrast ---
    # Use 5x5 uniform blur kernel
    kernel = torch.ones(1, 1, 5, 5, device=device, dtype=dtype) / 25.0
    Y_unsqueezed = Y.unsqueeze(1)  # [B, 1, H, W]

    # Reflect padding to avoid border artifacts
    Y_padded = F.pad(Y_unsqueezed, (2, 2, 2, 2), mode='reflect')
    Y_blur = F.conv2d(Y_padded, kernel)  # [B, 1, H, W]

    # Local contrast = absolute difference from local mean
    local_contrast = (Y_unsqueezed - Y_blur).abs().squeeze(1)  # [B, H, W]

    # --- Average over shadow regions ---
    # Avoid division by zero
    shadow_area = shadow_mask.sum()
    if shadow_area == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    shadow_detail_score = (local_contrast * shadow_mask).sum() / shadow_area
    return shadow_detail_score


if __name__ == '__main__':
    img = torch.rand((1, 3, 224, 224)).cuda()
    print(shadow_detail_torch(img))