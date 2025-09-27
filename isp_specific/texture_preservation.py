import torch
import torch.nn.functional as F
import cv2
import numpy as np


def texture_preservation_mean_torch(
        img: torch.Tensor,
        patch_size: int = 64,
        stride: int = 32
) -> torch.Tensor:
    """
    Quantify texture preservation using local standard deviation in patches.

    This metric evaluates how well the ISP preserves high-frequency details
    (e.g., skin texture, fabric, foliage) after noise reduction.

    Method:
    1. Convert to luminance
    2. Extract patches of size (patch_size x patch_size) with given stride
    3. Compute standard deviation of each patch (texture energy)
    4. Average over all patches

    Args:
        img (torch.Tensor): Input RGB image [B, 3, H, W], values in [0, 1]
        patch_size (int): Size of square patches. Default: 64
        stride (int): Stride between patches. Default: 32 (50% overlap)

    Returns:
        torch.Tensor: Scalar score representing average texture preservation.
                      Higher value = more texture retained.
                      Value is in intensity units (same as input luminance).

    Note:
        - This metric is commonly used to detect "plastic skin" effect in portraits.
        - For best results, use on images with actual texture content.
        - Typical values:
            * Over-smoothed: < 0.02
            * Natural texture: 0.02–0.08
            * High texture: > 0.08
        - Consider using Dead Leaves chart for standardized texture evaluation.
    """
    # Input validation
    if img.dim() != 4 or img.shape[1] != 3:
        raise ValueError("Input must be [B, 3, H, W]")
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive integers")
    if img.shape[2] < patch_size or img.shape[3] < patch_size:
        # Image too small for patches → return 0 (no texture measurable)
        return torch.tensor(0.0, device=img.device, dtype=img.dtype)

    # --- Convert to luminance (BT.601) ---
    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]  # [B, H, W]
    gray = gray.unsqueeze(1)  # [B, 1, H, W]

    # --- Extract patches using unfold ---
    # Result: [B, 1, nH, nW, patch_size, patch_size]
    patches = gray.unfold(2, patch_size, stride).unfold(3, patch_size, stride)

    # If no patches extracted (shouldn't happen due to check above, but safe)
    if patches.numel() == 0:
        return torch.tensor(0.0, device=img.device, dtype=img.dtype)

    # Reshape to [B, num_patches, patch_size*patch_size]
    patches_flat = patches.contiguous().view(patches.size(0), -1, patch_size * patch_size)

    # Compute standard deviation (use population std, not sample std)
    local_std = patches_flat.std(dim=-1, unbiased=False)  # [B, num_patches]

    # Return mean texture energy across all patches and batch
    return local_std.mean()


if __name__ == '__main__':
    img = torch.rand((1, 3, 224, 224)).cuda()
    print(texture_preservation_mean_torch(img))
