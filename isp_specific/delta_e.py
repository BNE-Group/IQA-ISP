import torch
import torch.nn.functional as F
import cv2
import numpy as np


def rgb_to_xyz_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB [0,1] to XYZ (D65 illuminant)."""
    if rgb.shape[1] != 3:
        raise ValueError("Expected [B, 3, H, W]")
    # Clamp to valid range to avoid negative XYZ
    rgb = torch.clamp(rgb, min=0.0, max=1.0)
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=rgb.dtype, device=rgb.device)
    return torch.einsum('ij,bjhw->bihw', M, rgb)


def xyz_to_lab_torch(xyz: torch.Tensor) -> torch.Tensor:
    """Convert XYZ to CIELAB (D65 white point)."""
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    xn = torch.clamp(xyz[:, 0] / Xn, min=0.0)
    yn = torch.clamp(xyz[:, 1] / Yn, min=0.0)
    zn = torch.clamp(xyz[:, 2] / Zn, min=0.0)

    delta = 6.0 / 29.0

    def f(t):
        return torch.where(
            t > delta ** 3,
            t ** (1.0 / 3.0),
            t / (3 * delta ** 2) + 4.0 / 29.0
        )

    fx, fy, fz = f(xn), f(yn), f(zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=1)


def delta_e_cie2000_from_rgb_torch(
        rgb1: torch.Tensor,
        rgb2: torch.Tensor,
        kL: float = 1.0,
        kC: float = 1.0,
        kH: float = 1.0
) -> torch.Tensor:
    """
    Compute CIEDE2000 color difference (ΔE00) between two RGB images.

    This is the industrial standard for color accuracy evaluation.

    Args:
        rgb1, rgb2 (torch.Tensor): RGB images [B, 3, H, W], values in [0, 1]
        kL, kC, kH (float): Weighting factors (default=1.0 for standard evaluation)

    Returns:
        torch.Tensor: Mean ΔE00 over all pixels (scalar).
                      Lower is better (ΔE00 < 2.0 is excellent).

    Reference:
        CIE Publication 142:2001, "Improvement to industrial colour-difference evaluation"
    """
    # Convert RGB to CIELAB
    xyz1 = rgb_to_xyz_torch(rgb1)
    xyz2 = rgb_to_xyz_torch(rgb2)
    lab1 = xyz_to_lab_torch(xyz1)
    lab2 = xyz_to_lab_torch(xyz2)

    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    # --- CIEDE2000 Formula ---
    # Mean values
    Lm = (L1 + L2) / 2
    am = (a1 + a2) / 2
    bm = (b1 + b2) / 2

    # Chroma
    C1 = torch.sqrt(a1 ** 2 + b1 ** 2 + 1e-12)
    C2 = torch.sqrt(a2 ** 2 + b2 ** 2 + 1e-12)
    Cm = (C1 + C2) / 2

    # Weighting function G
    G = 0.5 * (1 - torch.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7 + 1e-12)))

    # Adjusted a*
    ap1 = (1 + G) * a1
    ap2 = (1 + G) * a2

    # Adjusted Chroma
    Cp1 = torch.sqrt(ap1 ** 2 + b1 ** 2 + 1e-12)
    Cp2 = torch.sqrt(ap2 ** 2 + b2 ** 2 + 1e-12)
    Cpm = (Cp1 + Cp2) / 2

    # Hue angle (hp) in degrees [0, 360)
    def hue_angle(ap, b):
        hp = torch.atan2(b, ap) * 180.0 / torch.pi
        hp = torch.where(hp < 0, hp + 360.0, hp)
        return hp

    hp1 = hue_angle(ap1, b1)
    hp2 = hue_angle(ap2, b2)

    # Δh' (hue difference)
    dhp = hp2 - hp1
    # Wrap to [-180, 180]
    dhp = torch.where(dhp > 180, dhp - 360, dhp)
    dhp = torch.where(dhp < -180, dhp + 360, dhp)
    # If either chroma is zero, hue difference is undefined → set to 0
    dhp = torch.where((Cp1 < 1e-6) | (Cp2 < 1e-6), torch.zeros_like(dhp), dhp)

    # Mean hue (hpm)
    hpm = (hp1 + hp2) / 2
    # Handle circular mean when |hp1 - hp2| > 180
    hpm = torch.where(
        torch.abs(hp1 - hp2) > 180,
        (hpm + 180) % 360,
        hpm
    )
    # If either chroma is zero, set hpm to 0
    hpm = torch.where((Cp1 < 1e-6) | (Cp2 < 1e-6), torch.zeros_like(hpm), hpm)

    # T (hue rotation term)
    T = (1
         - 0.17 * torch.cos((hpm - 30) * torch.pi / 180)
         + 0.24 * torch.cos(2 * hpm * torch.pi / 180)
         + 0.32 * torch.cos((3 * hpm + 6) * torch.pi / 180)
         - 0.20 * torch.cos((4 * hpm - 63) * torch.pi / 180))

    # Δθ (hue shift)
    dtheta = 30 * torch.exp(-((hpm - 275) / 25) ** 2)

    # RC (chroma weighting)
    RC = 2 * torch.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7 + 1e-12))

    # SL, SC, SH (weighting factors)
    SL = 1 + (0.015 * (Lm - 50) ** 2) / torch.sqrt(20 + (Lm - 50) ** 2 + 1e-12)
    SC = 1 + 0.045 * Cpm
    SH = 1 + 0.015 * Cpm * T

    # RT (interaction term)
    RT = -torch.sin(2 * dtheta * torch.pi / 180) * RC

    # Differences
    dLp = L2 - L1
    dCp = Cp2 - Cp1
    dHp = 2 * torch.sqrt(Cp1 * Cp2 + 1e-12) * torch.sin(dhp * torch.pi / 360)

    # Final ΔE00
    dE2 = (dLp / (kL * SL)) ** 2 \
          + (dCp / (kC * SC)) ** 2 \
          + (dHp / (kH * SH)) ** 2 \
          + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))

    dE = torch.sqrt(torch.clamp(dE2, min=0))
    return dE.mean()

if __name__ == '__main__':
    img1 = torch.rand((1, 3, 224, 224)).cuda()
    img2 = torch.rand((1, 3, 224, 224)).cuda()

    print(delta_e_cie2000_from_rgb_torch(img1, img2))
