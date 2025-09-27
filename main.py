import torch
import cv2

from utils import analysis_isp_quality

if __name__ == '__main__':
    img1 = cv2.imread('Lenna.jpg')
    img2 = cv2.blur(img1, ksize=(5, 5))

    img1 = torch.Tensor(img1.transpose((2, 0, 1))).unsqueeze(0).cuda()
    img2 = torch.Tensor(img2.transpose((2, 0, 1))).unsqueeze(0).cuda()

    img1 = img1 / 255.0
    img2 = img2 / 255.0

    report = analysis_isp_quality(img1, img2)

    print('Image Quality Analysis')
    print('=========================')
    print('General Metrics:')
    for key in report['general']:
        print(f'    Metric: {key}, Val:{report['general'][key].item():.4f}')
    print('=========================')
    print('ISP Specific Metrics:')
    for key in report['isp_specific']:
        print(f'    Metric: {key}, Val:{report['isp_specific'][key].item():.4f}')