import torch
import torch.nn.functional as F
import cv2
import numpy as np


def chroma_noise_level_torch(img: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """
    Estimate chrominance noise level in an image.

    Chroma noise appears as random color speckles and is highly visible to human eyes.
    This function converts the image to YUV color space and computes the mean local
    variance in the U (Cb) and V (Cr) channels as a proxy for chroma noise energy.

    Args:
        img (torch.Tensor): Input RGB image tensor of shape [B, 3, H, W].
                            Expected to be normalized to [0, 1] range.
        window_size (int): Size of the local window for variance estimation.
                           Must be odd. Default: 7.

    Returns:
        torch.Tensor: Scalar tensor representing average chroma noise level.
                      Higher value = more chroma noise.
                      Value is in squared intensity units (e.g., if input in [0,1], output in [0,1]).

    Note:
        - Uses BT.601 coefficients for YUV conversion (standard for SDTV, commonly used in ISP).
        - Local variance is computed via: Var(X) = E[X²] - (E[X])²
        - Negative variances due to floating-point error are clamped to zero.
        - This metric may overestimate noise in textured/colorful regions.
    """
    assert img.dim() == 4 and img.shape[1] == 3, "Input must be [B, 3, H, W]"
    assert window_size % 2 == 1, "window_size must be odd"

    R, G, B = img[:, 0], img[:, 1], img[:, 2]

    # --- RGB to YUV (BT.601 standard) ---
    # Luma (Y)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    # Chroma (Cb, Cr) - using precise BT.601 coefficients
    U = -0.169 * R - 0.331 * G + 0.5 * B  # Cb
    V = 0.5 * R - 0.419 * G - 0.081 * B  # Cr

    # Add channel dimension for conv2d: [B, 1, H, W]
    U = U.unsqueeze(1)
    V = V.unsqueeze(1)

    # Padding to maintain spatial size after convolution
    pad = window_size // 2
    U_pad = F.pad(U, (pad, pad, pad, pad), mode='reflect')
    V_pad = F.pad(V, (pad, pad, pad, pad), mode='reflect')

    # Uniform averaging kernel for local mean
    kernel = torch.ones(1, 1, window_size, window_size, device=img.device, dtype=img.dtype) / (window_size ** 2)

    # Compute local mean
    U_mean = F.conv2d(U_pad, kernel)
    V_mean = F.conv2d(V_pad, kernel)

    # Compute local variance: Var = E[X^2] - (E[X])^2
    U_var = F.conv2d(U_pad ** 2, kernel) - U_mean ** 2
    V_var = F.conv2d(V_pad ** 2, kernel) - V_mean ** 2

    # Clamp negative variances (caused by floating-point precision)
    U_var = torch.clamp(U_var, min=0.0)
    V_var = torch.clamp(V_var, min=0.0)

    # Average chroma noise across U and V channels and all pixels
    chroma_noise = (U_var.mean() + V_var.mean()) / 2.0
    return chroma_noise


if __name__ == '__main__':
    img = torch.rand((1, 3, 224, 224)).cuda()
    print(chroma_noise_level_torch(img))
