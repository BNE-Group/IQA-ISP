import torch
import torch.nn.functional as F
import cv2
import numpy as np


def color_fringing_torch(
    img: torch.Tensor,
    edge_threshold: float = 0.1,
    r_b_threshold: float = 0.3,
    g_ratio_threshold: float = 0.7
) -> torch.Tensor:
    """
    Detect purple fringing artifacts in an image.

    Purple fringing appears as magenta/purple halos along high-contrast edges,
    typically on the bright side of dark-to-bright transitions.

    This function:
    1. Detects strong luminance edges
    2. Identifies pixels with high R/B and low G (purple signature)
    3. Ensures these pixels are near edges AND on the bright side

    Args:
        img (torch.Tensor): Input RGB image [B, 3, H, W], values in [0, 1]
        edge_threshold (float): Minimum gradient magnitude to be considered an edge
        r_b_threshold (float): Minimum R and B intensity to be considered "bright"
        g_ratio_threshold (float): Maximum G / min(R, B) ratio for purple classification

    Returns:
        torch.Tensor: Scalar score in [0, 1]. Higher = more severe purple fringing.

    Note:
        - This is a heuristic detector; may have false positives on purple objects
        - For best results, use on images with high dynamic range (e.g., backlit scenes)
    """
    assert img.dim() == 4 and img.shape[1] == 3, "Input must be [B, 3, H, W]"
    device, dtype = img.device, img.dtype

    R, G, B = img[:, 0], img[:, 1], img[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B  # Luminance [B, H, W]

    # --- Compute luminance gradients (Scharr operator) ---
    # Scharr kernels (more rotationally symmetric than Sobel)
    scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                           dtype=dtype, device=device).view(1, 1, 3, 3) / 32.0
    scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
                           dtype=dtype, device=device).view(1, 1, 3, 3) / 32.0

    Y_pad = F.pad(Y.unsqueeze(1), (1, 1, 1, 1), mode='reflect')
    gx = F.conv2d(Y_pad, scharr_x)
    gy = F.conv2d(Y_pad, scharr_y)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2).squeeze(1)  # [B, H, W]

    # --- Detect strong edges ---
    strong_edges = grad_mag > edge_threshold  # [B, H, W]

    # --- Identify purple pixels ---
    # Condition: R and B are high, G is low relative to R/B
    min_rb = torch.min(R, B)
    purple_mask = (R > r_b_threshold) & (B > r_b_threshold) & (G < g_ratio_threshold * min_rb)

    # --- Heuristic: purple fringing appears on BRIGHT side of edges ---
    # Approximate bright side as: pixels with luminance > local average
    # Compute local mean luminance (5x5 window)
    local_mean = F.avg_pool2d(F.pad(Y.unsqueeze(1), (2,2,2,2), mode='reflect'), 5, stride=1).squeeze(1)
    bright_side = Y > local_mean  # [B, H, W]

    # --- Combine conditions ---
    # Purple fringing = purple pixels that are (near edges AND on bright side)
    edge_dilated = F.max_pool2d(
        strong_edges.float().unsqueeze(1),
        kernel_size=3, stride=1, padding=1
    ).squeeze(1)  # Dilate to cover fringe width

    fringing_mask = purple_mask & (edge_dilated > 0) & bright_side
    fringing_score = fringing_mask.float().mean()

    return fringing_score


if __name__ == '__main__':
    img = torch.rand((1, 3, 224, 224)).cuda()
    print(color_fringing_torch(img))