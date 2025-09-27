import torch
import torch.nn.functional as F
import cv2
import numpy as np


def effective_dynamic_range_snr_torch(
        img: torch.Tensor,
        window_size: int = 7,
        snr_threshold: float = 1.0,
        dark_threshold: float = 0.01,
        bright_threshold: float = 0.99
) -> torch.Tensor:
    """
    Estimate Effective Dynamic Range (EDR) in Exposure Values (EV) using SNR criterion.

    EDR = log2(Y_max / Y_min), where:
      - Y_max: maximum usable luminance (non-clipped highlights)
      - Y_min: minimum luminance with SNR > snr_threshold

    This metric evaluates how well the ISP preserves details in both shadows and highlights.

    Args:
        img (torch.Tensor): Input RGB image [B, 3, H, W], values in [0, 1]
        window_size (int): Window size for local noise estimation. Default: 7
        snr_threshold (float): Minimum SNR for a pixel to be considered "usable". Default: 1.0
        dark_threshold (float): Minimum luminance to consider (avoids pure black). Default: 0.01
        bright_threshold (float): Maximum luminance to exclude clipped highlights. Default: 0.99

    Returns:
        torch.Tensor: Scalar EDR in EV. Returns NaN if no usable pixels found.

    Note:
        - This is a simplified SNR estimator; for production, consider:
          * Edge-aware noise estimation
          * Multi-scale analysis
        - Typical EDR values:
          * Standard sRGB: 6-8 EV
          * Good HDR: 9-12 EV
    """
    assert img.dim() == 4 and img.shape[1] == 3, "Input must be [B, 3, H, W]"
    device, dtype = img.device, img.dtype

    # --- Convert to luminance (BT.709) ---
    Y = 0.2126 * img[:, 0] + 0.7152 * img[:, 1] + 0.0722 * img[:, 2]  # [B, H, W]
    Y = Y.unsqueeze(1)  # [B, 1, H, W]

    # --- Estimate noise via local variance (simplified) ---
    pad = window_size // 2
    Y_pad = F.pad(Y, (pad, pad, pad, pad), mode='reflect')
    kernel = torch.ones(1, 1, window_size, window_size, device=device, dtype=dtype) / (window_size ** 2)

    mean_Y = F.conv2d(Y_pad, kernel)
    var_Y = F.conv2d(Y_pad ** 2, kernel) - mean_Y ** 2
    noise_est = torch.sqrt(torch.clamp(var_Y, min=0))  # [B, 1, H, W]

    # --- Compute SNR and usable mask ---
    snr = Y / (noise_est + 1e-8)  # Avoid division by zero
    usable = (snr > snr_threshold) & (Y > dark_threshold)  # [B, 1, H, W]

    # --- Find Y_max: max luminance in non-clipped region ---
    bright_mask = Y < bright_threshold  # [B, 1, H, W]

    # Handle case where no pixels are below bright_threshold
    if not bright_mask.any():
        Y_max = Y.view(Y.size(0), -1).max(dim=1).values
    else:
        Y_masked = torch.where(bright_mask, Y, torch.tensor(-1.0, device=device, dtype=dtype))
        Y_max = Y_masked.view(Y.size(0), -1).max(dim=1).values
        Y_max = torch.clamp(Y_max, min=1e-6)  # Ensure positive

    # --- Find Y_min: min luminance in usable region ---
    if not usable.any():
        # No usable pixels → return NaN
        return torch.tensor(float('nan'), device=device, dtype=dtype)

    Y_usable = torch.where(usable, Y, torch.tensor(1e6, device=device, dtype=dtype))
    Y_min = Y_usable.view(Y.size(0), -1).min(dim=1).values
    Y_min = torch.clamp(Y_min, min=1e-6)

    # --- Compute EDR in EV ---
    # Ensure Y_max >= Y_min to avoid negative EV
    edr_ev = torch.log2(torch.clamp(Y_max / Y_min, min=1.0))
    return edr_ev.mean()


if __name__ == '__main__':
    img = torch.rand((1, 3, 224, 224)).cuda()
    print(effective_dynamic_range_snr_torch(img))
