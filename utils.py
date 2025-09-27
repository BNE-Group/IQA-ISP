import torch

from general import *
from isp_specific import *

def analysis_isp_quality(img1: torch.Tensor, img2=None) -> dict:

    report = {
        'general': {},
        'isp_specific': {}
    }

    # general
    if img2 is not None:
        report['general']['psnr'] = iqa_psnr(img1, img2)
        report['general']['ssim'] = iqa_ssim(img1, img2)
        report['general']['lpips'] = iqa_lpips(img1, img2)

    # isp specific
    if img2 is not None:
        report['isp_specific']['deltaE'] = iqa_deltaE(img1, img2)

    # details
    report['isp_specific']['blocking'] = iqa_blocking_artifacts(img1)
    report['isp_specific']['shadow_detail'] = iqa_shadow_detail(img1)
    report['isp_specific']['sharpness'] = iqa_tenengrad_sharpness(img1)
    report['isp_specific']['texture_preservation'] = iqa_texture_preservation(img1)
    report['isp_specific']['watercolor'] = iqa_watercolor(img1)

    # noise
    report['isp_specific']['noise_chroma'] = iqa_chroma_noise(img1)
    report['isp_specific']['noise_luma'] = iqa_luma_noise(img1)
    report['isp_specific']['local_var_noise'] = iqa_local_variance_noise(img1)

    # color
    report['isp_specific']['color_fringing'] = iqa_color_fringing(img1)
    report['isp_specific']['colorfulness'] = iqa_colorfulness(img1)

    # light
    report['isp_specific']['effective_dr'] = iqa_effective_dr(img1)

    return report
