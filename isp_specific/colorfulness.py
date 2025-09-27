import torch
import torch.nn.functional as F
import cv2
import numpy as np


def colorfulness_lab_perceptual(img: torch.Tensor) -> torch.Tensor:
    """
    Estimate perceptual colorfulness using CIELAB color space.

    Based on the method of Hasler & Süsstrunk (2003), adapted to use CIELAB
    for better perceptual accuracy. Colorfulness is defined as:
        Colorfulness = σ_C + 0.3 * μ_C
    where C = √(a*² + b*²) is the chroma per pixel.

    Args:
        img (torch.Tensor): Input RGB image tensor of shape [B, 3, H, W].
                            Values must be in [0, 1] range (sRGB).

    Returns:
        torch.Tensor: Scalar tensor representing average colorfulness.
                      Higher value = more colorful image.

    Note:
        - This metric correlates well with human perception of color richness.
        - Typical values:
            * Grayscale: ~0
            * Natural scenes: 20–40
            * Over-saturated: >50
    """
    assert img.dim() == 4 and img.shape[1] == 3, "Input must be [B, 3, H, W]"

    # Clamp input to [0, 1] to avoid invalid XYZ values
    img = torch.clamp(img, min=0.0, max=1.0)
    device, dtype = img.device, img.dtype

    # --- RGB to XYZ (sRGB, D65) ---
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=dtype, device=device)
    xyz = torch.einsum('ij,bjhw->bihw', M, img)

    # --- XYZ to CIELAB (D65 white point) ---
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    xn = xyz[:, 0] / Xn
    yn = xyz[:, 1] / Yn
    zn = xyz[:, 2] / Zn

    # Clamp to avoid negative values (can occur due to RGB clamping)
    xn = torch.clamp(xn, min=0.0)
    yn = torch.clamp(yn, min=0.0)
    zn = torch.clamp(zn, min=0.0)

    # Non-linear compression function
    delta = 6.0 / 29.0

    def f(t):
        return torch.where(
            t > delta ** 3,
            t ** (1.0 / 3.0),
            t / (3 * delta ** 2) + 4.0 / 29.0
        )

    fx, fy, fz = f(xn), f(yn), f(zn)
    a = 500 * (fx - fy)  # [B, H, W]
    b = 200 * (fy - fz)  # [B, H, W]

    # --- Compute chroma and colorfulness ---
    chroma = torch.sqrt(a ** 2 + b ** 2 + 1e-8)  # [B, H, W]

    # Use unbiased=False (population std, not sample std)
    mu = chroma.mean(dim=[1, 2])  # [B]
    sigma = chroma.std(dim=[1, 2], unbiased=False)  # [B]

    colorfulness = sigma + 0.3 * mu  # [B]
    return colorfulness.mean()  # scalar

if __name__ == '__main__':
    img1 = torch.rand((1, 3, 224, 224)).cuda()

    print(colorfulness_lab_perceptual(img1))
