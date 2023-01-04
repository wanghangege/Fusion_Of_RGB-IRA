'''
    交叉双边滤波器(CBF)
    paper: 《Image fusion based on pixel significance using cross bilateral filter》
    author(modify, not original): Benwu Wang
    date: 2022/11/11
'''

import numpy as np
import cv2
import argparse
import math
import csv
import os
from fusionone.fusiononemmain import psnr_of_PSNR
from fusionone.fusiononemmain import calculate_ssim_Gaussion, SSIM_GUAN
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
cov_wsize = 5
sigmas = 1.8
sigmar = 25
ksize = 11

def gaussian_kernel_2d_opencv(kernel_size = 11,sigma = 1.8):
    kx = cv2.getGaussianKernel(kernel_size,sigma)
    ky = cv2.getGaussianKernel(kernel_size,sigma)
    return np.multiply(kx,np.transpose(ky))

def bilateralFilterEx(img_r, img_v):
    #edge solved
    win_size = ksize//2
    img_r_copy = None
    img_v_copy = None
    img_r_copy = cv2.copyTo(img_r, None)
    img_v_copy = cv2.copyTo(img_v, None)
    img_r_cbf = np.ones_like(img_r, dtype=np.float)
    img_v_cbf = np.ones_like(img_r, dtype=np.float)
    img_r_copy = np.pad(img_r_copy, (win_size, win_size), 'reflect')
    img_v_copy = np.pad(img_v_copy, (win_size, win_size), 'reflect')
    gk = gaussian_kernel_2d_opencv()
    for i in range(win_size, win_size+img_r.shape[0]):
        for j in range(win_size, win_size+img_r.shape[1]):
            sumr1 = 0.
            sumr2 = 0.
            sumv1 = 0.
            sumv2 = 0.
            img_r_cdis = img_r_copy[i-win_size:i+win_size+1, j-win_size:j+win_size+1] *1.0- img_r_copy[i,j]*1.0
            img_v_cdis = img_v_copy[i-win_size:i+win_size+1, j-win_size:j+win_size+1] *1.0- img_v_copy[i,j]*1.0
            sumr1 = np.sum(np.exp(-img_v_cdis*img_v_cdis) *gk/ (2*sigmar*sigmar) )
            sumv1 = np.sum(np.exp(-img_r_cdis*img_r_cdis) *gk/ (2*sigmar*sigmar) )
            sumr2 = np.sum(np.exp(-img_v_cdis*img_v_cdis) *gk*img_r_copy[i-win_size:i+win_size+1, j-win_size:j+win_size+1] *1.0/ (2*sigmar*sigmar) )
            sumv2 = np.sum(np.exp(-img_r_cdis*img_r_cdis) *gk*img_v_copy[i-win_size:i+win_size+1, j-win_size:j+win_size+1] *1.0/ (2*sigmar*sigmar) )
            img_r_cbf[i-win_size,j-win_size] = sumr2 / sumr1
            img_v_cbf[i-win_size,j-win_size] = sumv2 / sumv1
    return (img_r*1. - img_r_cbf, img_v*1. - img_v_cbf)

def CBF_WEIGHTS(img_r_d, img_v_d):
    win_size = cov_wsize // 2
    img_r_weights = np.ones_like(img_r_d, dtype=np.float)
    img_v_weights= np.ones_like(img_v_d, dtype=np.float)
    img_r_d_pad = np.pad(img_r_d, (win_size, win_size), 'reflect')
    img_v_d_pad = np.pad(img_v_d, (win_size, win_size), 'reflect')
    for i in range(win_size, win_size+img_r_d.shape[0]):
        for j in range(win_size, win_size+img_r_d.shape[1]):
            npt_r = img_r_d_pad[i-win_size:i+win_size+1, j-win_size:j+win_size+1]
            npt_v = img_v_d_pad[i-win_size:i+win_size+1, j-win_size:j+win_size+1]
            npt_r_V = npt_r - np.mean(npt_r, axis=0)
            npt_r_V = npt_r_V*npt_r_V.transpose()
            npt_r_H = npt_r.transpose() - np.mean(npt_r, axis=1)
            npt_r_H = npt_r_H*npt_r_H.transpose()
            npt_v_V = npt_v - np.mean(npt_v, axis=0)
            npt_v_V = npt_v_V*npt_v_V.transpose()
            npt_v_H = npt_v.transpose() - np.mean(npt_v, axis=1)
            npt_v_H = npt_v_H*npt_v_H.transpose()
            img_r_weights[i-win_size,j-win_size] = np.trace(npt_r_H) + np.trace(npt_r_V)
            img_v_weights[i-win_size,j-win_size] = np.trace(npt_v_H) + np.trace(npt_v_V)
    return img_r_weights, img_v_weights

def CBF_GRAY(img_r, img_v):
    img_r_d, img_v_d = bilateralFilterEx(img_r, img_v)
    img_r_weights, img_v_weights = CBF_WEIGHTS(img_r_d, img_v_d)
    img_fused =(img_r*1. * img_r_weights + img_v*1.*img_v_weights) /(img_r_weights+img_v_weights)
    img_fused = cv2.convertScaleAbs(img_fused)
    return img_fused

def CBF_RGB(img_r, img_v):
    img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    img_v_gray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
    return CBF_GRAY(img_r_gray, img_v_gray)

def CBF(_rpath, _vpath, fused_img_RGB):
    # img_r = cv2.imread(_rpath)
    # img_v = cv2.imread(_vpath)
    img_r = _rpath
    img_v = _vpath
    if not isinstance(img_r, np.ndarray) :
        print('img_r is null')
        return
    if not isinstance(img_v, np.ndarray) :
        print('img_v is null')
        return
    if img_r.shape[0] != img_v.shape[0]  or img_r.shape[1] != img_v.shape[1]:
        print('size is not equal')
        return
    fused_img = None
    if len(img_r.shape)  < 3 or img_r.shape[2] ==1:
        if len(img_v.shape)  < 3 or img_v.shape[-1] ==1:
            fused_img = CBF_GRAY(img_r, img_v)
        else:
            img_v_gray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
            fused_img = CBF_GRAY(img_r, img_v)
    else:
        if len(img_v.shape)  < 3 or img_v.shape[-1] ==1:
            img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            fused_img = CBF_GRAY(img_r_gray, img_v)
        else:
            fused_img = CBF_RGB(img_r, img_v)
    fused_img_RGB[:, :, 0] = fused_img
    fused_img_RGB[:, :, 1] = fused_img
    fused_img_RGB[:, :, 2] = fused_img

    return fused_img_RGB
    # cv2.imshow('fused image', fused_img)
    # cv2.imwrite("fused_image.jpg", fused_img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--IR", default=r'E:\SIMpro\pro_infofusion\data\Ir\00061.png', help="path to IR image", required=False)
    # parser.add_argument("--VIS", default=r'E:\SIMpro\pro_infofusion\data\Vis\00061.png', help="path to IR image", required=False)
    # a = parser.parse_args()
    path_rgb = r"E:\SIMpro\pro_infofusion\fusionone\infra"
    path_ira = r"E:\SIMpro\pro_infofusion\fusionone\visrgb"
    csv_file =  open('results_CBF.csv', 'w', encoding='gbk', newline="")
    csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(["image name", "pnsr value", "official ssim value", "my ssim value"])
    csv_writer.writerow(["图像名称", "PNSR值", "SSIM值", "SSIM值（自定义）", "my value"])
    psnr_v = []
    ssim_v1 = []
    ssim_v2 = []
    ssim_v3 = []
    psnr_vf_v, psnr_if_v = [], []
    for img_ids in os.listdir(path_rgb):
        img_id = img_ids.split('.')[0]
        print(img_id)
        rgb_id = cv2.imread(os.path.join(path_rgb, img_id + '.png'))
        ira_id = cv2.imread(os.path.join(path_ira, img_id + '.png'))
        fused_img_RGB = np.zeros_like(rgb_id)
        fus_img = CBF(ira_id, rgb_id, fused_img_RGB)
        # fus_img = ira_id
        cv2.imwrite('E://SIMpro//pro_infofusion//fusion_CBF//results//' + '{}_fusion.png'.format(img_id), fus_img)
        visrgb = rgb_id
        infra = ira_id
        fusion_last = fus_img
        psnr = psnr_of_PSNR(visrgb, infra, fusion_last)
        psnr_vf = compare_psnr(visrgb, fusion_last)
        psnr_if = compare_psnr(infra, fusion_last)
        ssim2 = SSIM_GUAN(visrgb, infra, fusion_last)
        ssim3 = calculate_ssim_Gaussion(visrgb, fusion_last) + calculate_ssim_Gaussion(infra, fusion_last)
        print('psnr, psnr_vf, psnr_if, ssim2, ssim3 value: ', psnr, psnr_vf, psnr_if, ssim2, ssim3)

        info_csv = [img_id+ '.png', psnr, ssim2, ssim3]
        csv_writer.writerow(info_csv)

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
    csv_file.close()