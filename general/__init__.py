from general.lpips import lpips_torch as iqa_lpips
from general.ssim import ssim_torch as iqa_ssim
from general.psnr import psnr_torch as iqa_psnr

__all__ = [
    'iqa_lpips', 'iqa_psnr', 'iqa_ssim'
]
