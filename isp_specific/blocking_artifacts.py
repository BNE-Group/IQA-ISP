import torch
import torch.nn.functional as F
import cv2
import numpy as np


def blocking_artifacts_torch(img: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """
    Quantify blocking artifacts by measuring discontinuities at block boundaries.

    Block artifacts appear as visible seams between processing blocks (e.g., 8x8 JPEG blocks).
    This function computes the mean absolute difference (MAD) across vertical and horizontal
    block boundaries.

    Args:
        img (torch.Tensor): Input image tensor of shape [B, C, H, W], values in any range.
                            Expected to be on the same device as computation.
        block_size (int): Size of processing block (e.g., 8 for JPEG). Default: 8.

    Returns:
        torch.Tensor: Scalar tensor representing blocking artifact strength.
                      Higher value = more severe blocking.
                      Returns NaN if image is too small to contain block boundaries.

    Note:
        - Only considers full blocks (ignores incomplete blocks at image edges).
        - Measures discontinuity between column x=block_size*k-1 and x=block_size*k (vertical boundaries),
          and between row y=block_size*k-1 and y=block_size*k (horizontal boundaries).
    """

    B, C, H, W = img.shape
    device = img.device
    dtype = img.dtype

    # Compute the largest coordinate that aligns with block grid
    max_x = (W // block_size) * block_size  # e.g., W=640 → max_x=640; W=645 → max_x=640
    max_y = (H // block_size) * block_size

    # If image has fewer than 2 full blocks in either dimension, no block boundaries exist
    if max_x < 2 * block_size or max_y < 2 * block_size:
        # Return NaN to indicate undefined metric (alternatively, return 0.0 if preferred)
        return torch.tensor(float('nan'), device=device, dtype=dtype)

    # Vertical block boundaries occur at columns: block_size, 2*block_size, ..., max_x - block_size
    vert_cols = torch.arange(block_size, max_x, block_size, device=device)  # [N_v]
    # Horizontal block boundaries occur at rows: block_size, 2*block_size, ..., max_y - block_size
    hori_rows = torch.arange(block_size, max_y, block_size, device=device)  # [N_h]

    # Compute vertical discontinuities: |I[:, :, :, x] - I[:, :, :, x-1]| at each boundary x
    if len(vert_cols) > 0:
        left_pixels = img[:, :, :, vert_cols - 1]  # [B, C, H, N_v]
        right_pixels = img[:, :, :, vert_cols]  # [B, C, H, N_v]
        vert_diff = torch.abs(right_pixels - left_pixels).mean()
    else:
        vert_diff = torch.tensor(0.0, device=device, dtype=dtype)

    # Compute horizontal discontinuities: |I[:, :, y, :] - I[:, :, y-1, :]| at each boundary y
    if len(hori_rows) > 0:
        top_pixels = img[:, :, hori_rows - 1, :]  # [B, C, N_h, W]
        bottom_pixels = img[:, :, hori_rows, :]  # [B, C, N_h, W]
        hori_diff = torch.abs(bottom_pixels - top_pixels).mean()
    else:
        hori_diff = torch.tensor(0.0, device=device, dtype=dtype)

    # Average vertical and horizontal blocking strengths
    blocking_score = (vert_diff + hori_diff) / 2.0
    return blocking_score


if __name__ == '__main__':
    img = torch.rand((1, 3, 224, 224)).cuda()
    print(blocking_artifacts_torch(img))