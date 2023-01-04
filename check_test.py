import numpy as np
import cv2
import argparse
import csv
import os
from fusionone.fusiononemmain import psnr_of_PSNR
from fusionone.fusiononemmain import calculate_ssim_Gaussion, SSIM_GUAN
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import os
vis_dir = r'E:\SIMpro\pro_infofusion\data\Vis'
ira_dir = r'E:\SIMpro\pro_infofusion\data\Ir'
path_test = r'E:\SIMpro\pro_infofusion\result'
psnr_v = []
ssim_v1 = []
ssim_v2 = []
ssim_v3 = []
psnr_vf_v, psnr_if_v = [], []
for png in os.listdir(path_test):
    fusion_name = png[:5] + '.png'
    print(fusion_name)
    fus_img = os.path.join(path_test, png)
    rgb_img = os.path.join(vis_dir, fusion_name)
    ira_img = os.path.join(ira_dir, fusion_name)

    visrgb = cv2.imread(rgb_img)
    infra = cv2.imread(ira_img)
    fusion_last = cv2.imread(fus_img)
    psnr = psnr_of_PSNR(visrgb, infra, fusion_last)
    psnr_vf = compare_psnr(visrgb, fusion_last)
    psnr_if = compare_psnr(infra, fusion_last)
    ssim2 = SSIM_GUAN(visrgb, infra, fusion_last)
    ssim3 = calculate_ssim_Gaussion(visrgb, fusion_last) + calculate_ssim_Gaussion(infra, fusion_last)
    print('psnr, psnr_vf, psnr_if, ssim2, ssim3 value: ', psnr, psnr_vf, psnr_if, ssim2, ssim3)

    psnr_v.append(psnr)
    ssim_v2.append(ssim2)
    ssim_v3.append(ssim3)
    psnr_vf_v.append(psnr_vf)
    psnr_if_v.append(psnr_if)

print('Finished testing')
print('The evaluted vale:')
print('psnr_v : ', np.mean(psnr_v))
print('ssim_v2: ', np.mean(ssim_v2))
print('ssim_v3: ', np.mean(ssim_v3))
print('psnr_vf_v: ', np.mean(psnr_vf_v))
print('psnr_if_v: ', np.mean(psnr_if_v))

