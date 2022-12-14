'''
    VSM是visual saliency map(视觉显著映射)的缩写，WLS是weighted least square(最小加权二乘法)
    paper: 《Infrared and visible image fusion based on visual saliency map and weighted least square optimization》
    author(modify, not original): Benwu Wang
    date: 2022/11/11
'''
import numpy as np
import cv2
import argparse
import math
from scipy.sparse import csr_matrix
import csv
import os
from fusionone.fusiononemmain import psnr_of_PSNR
from fusionone.fusiononemmain import calculate_ssim_Gaussion, SSIM_GUAN
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

ITERATION = 4
SIGMA_S = 2.
SIGMA_R = 0.05
WLS_LAMBDA = 0.01


def guidedFilter(img_i, img_p, r, eps):
    wsize = int(2 * r) + 1
    meanI = cv2.boxFilter(img_i, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    meanP = cv2.boxFilter(img_p, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    corrI = cv2.boxFilter(img_i * img_i, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    corrIP = cv2.boxFilter(img_i * img_p, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    varI = corrI - meanI * meanI
    covIP = corrIP - meanI * meanP
    a = covIP / (varI + eps)
    b = meanP - a * meanI
    meanA = cv2.boxFilter(a, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    meanB = cv2.boxFilter(b, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    q = meanA * img_i + meanB
    return q


def RGF(img_r, img_v, sigma_s, sigma_r):
    img_r_gs = cv2.GaussianBlur(img_r, (int(3 * sigma_s) * 2 + 1, int(3 * sigma_s) * 2 + 1), sigma_s)
    img_v_gs = cv2.GaussianBlur(img_v, (int(3 * sigma_s) * 2 + 1, int(3 * sigma_s) * 2 + 1), sigma_s)
    for i in range(ITERATION):
        img_r_gf = guidedFilter(img_r_gs, img_r, sigma_s, sigma_r * sigma_r)
        img_v_gf = guidedFilter(img_v_gs, img_v, sigma_s, sigma_r * sigma_r)
        img_r_gs = img_r_gf
        img_v_gs = img_v_gf
    return img_r_gf, img_v_gf


def SOLVE_OPTIMAL(M, img_r_d, img_v_d, wls_lambda):
    m, n = img_r_d.shape
    small_number = 0.0001
    img_r_d_blur = np.abs(cv2.blur(img_r_d, (7, 7)))
    img_r_d_blur = 1 / (img_r_d_blur + small_number)
    _row = np.arange(0, m * n)
    _col = np.arange(0, m * n)
    _data_A = img_r_d_blur.reshape(m * n)
    _data_M = M.reshape(m * n)
    _data_d2 = img_v_d.reshape(m * n)
    _data_I = np.array([1] * (m * n))
    D = (_data_M + wls_lambda * _data_A * _data_d2) / (_data_I + wls_lambda * _data_A)
    return D.reshape([m, n])


def VSMWLS_GRAY(img_r, img_v):
    bases_r = []
    bases_v = []
    details_r = []
    details_v = []
    img_r_copy = img_r * 1.0 / 255
    img_v_copy = img_v * 1.0 / 255
    bases_r.append(img_r_copy)
    bases_v.append(img_v_copy)
    sigma_s = SIGMA_S
    img_r_rgf = None
    img_v_rgf = None
    for i in range(ITERATION - 1):
        img_r_rgf, img_v_rgf = RGF(img_r_copy, img_v_copy, sigma_s, SIGMA_R)
        bases_r.append(img_r_rgf)
        bases_v.append(img_v_rgf)
        details_r.append(bases_r[i] - bases_r[i + 1])
        details_v.append(bases_v[i] - bases_v[i + 1])
        sigma_s *= 2
    sigma_s = 2
    img_r_base = cv2.GaussianBlur(bases_r[ITERATION - 1], (int(3 * sigma_s) * 2 + 1, int(3 * sigma_s) * 2 + 1), sigma_s)
    img_v_base = cv2.GaussianBlur(bases_v[ITERATION - 1], (int(3 * sigma_s) * 2 + 1, int(3 * sigma_s) * 2 + 1), sigma_s)
    details_r.append(bases_r[ITERATION - 1] - img_r_base)
    details_v.append(bases_v[ITERATION - 1] - img_v_base)
    img_r_hist = cv2.calcHist([img_r], [0], None, [256], [0, 255])
    img_v_hist = cv2.calcHist([img_v], [0], None, [256], [0, 255])
    img_r_hist_tab = np.zeros(256, np.int)
    img_v_hist_tab = np.zeros(256, np.int)
    for i in range(256):
        for j in range(256):
            img_r_hist_tab[i] = img_r_hist_tab[i] + img_r_hist[j] * math.fabs(i - j)
            img_v_hist_tab[i] = img_v_hist_tab[i] + img_v_hist[j] * math.fabs(i - j)
    img_r_base_weights = img_r.astype(np.int)
    img_v_base_weights = img_v.astype(np.int)
    for i in range(256):
        img_r_base_weights[img_r_base_weights == i] = img_r_hist_tab[i]
        img_v_base_weights[img_v_base_weights == i] = img_v_hist_tab[i]
    img_r_b_weights = cv2.normalize(img_r_base_weights * 1., None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    img_v_b_weights = cv2.normalize(img_v_base_weights * 1., None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    img_fused_base = img_r_base * (0.5 + (img_r_b_weights - img_v_b_weights) / 2) + img_v_base * (
                0.5 + (img_v_b_weights - img_r_b_weights) / 2)
    w = int(3 * sigma_s)
    C_0 = (details_v[0] < details_r[0]).astype(np.float)
    C_0 = cv2.GaussianBlur(C_0, (2 * w + 1, 2 * w + 1), sigma_s)
    img_fused_detail = C_0 * details_r[0] + (1 - C_0) * details_v[0]
    for i in range(ITERATION - 1, 0, -1):
        C_0 = (details_v[i] < details_r[i]).astype(np.float)
        C_0 = cv2.GaussianBlur(C_0, (2 * w + 1, 2 * w + 1), sigma_s)
        M = C_0 * details_r[i] + (1 - C_0) * details_v[i]
        fused_di = SOLVE_OPTIMAL(M, details_r[i], details_v[i], WLS_LAMBDA)
        img_fused_detail += fused_di
    img_fused = img_fused_base + img_fused_detail
    img_fused = cv2.normalize(img_fused, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.convertScaleAbs(img_fused)


def VSMWLS_RGB(img_r, img_v):
    fused_img = np.ones_like(img_r)
    r_R = img_r[:, :, 2]
    v_R = img_v[:, :, 2]
    r_G = img_r[:, :, 1]
    v_G = img_v[:, :, 1]
    r_B = img_r[:, :, 0]
    v_B = img_v[:, :, 0]
    fused_R = VSMWLS_GRAY(r_R, v_R)
    fused_G = VSMWLS_GRAY(r_G, v_G)
    fused_B = VSMWLS_GRAY(r_B, v_B)
    fused_img[:, :, 2] = fused_R
    fused_img[:, :, 1] = fused_G
    fused_img[:, :, 0] = fused_B
    return fused_img


def VSMWLS(_rpath, _vpath):
    # img_r = cv2.imread(_rpath)
    # img_v = cv2.imread(_vpath)
    img_r = _rpath
    img_v = _vpath
    if not isinstance(img_r, np.ndarray):
        print('img_r is null')
        return
    if not isinstance(img_v, np.ndarray):
        print('img_v is null')
        return
    if img_r.shape[0] != img_v.shape[0] or img_r.shape[1] != img_v.shape[1]:
        print('size is not equal')
        return
    fused_img = None
    if len(img_r.shape) < 3 or img_r.shape[2] == 1:
        if len(img_v.shape) < 3 or img_v.shape[-1] == 1:
            fused_img = VSMWLS_GRAY(img_r, img_v)
        else:
            img_v_gray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
            fused_img = VSMWLS_GRAY(img_r, img_v)
    else:
        if len(img_v.shape) < 3 or img_v.shape[-1] == 1:
            img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            fused_img = VSMWLS_GRAY(img_r_gray, img_v)
        else:
            fused_img = VSMWLS_RGB(img_r, img_v)
    return fused_img
    # cv2.imshow('fused image', fused_img)
    # cv2.imwrite("fused_image_vsmwls.jpg", fused_img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--r", default=r'E:\SIMpro\pro_infofusion\data\Ir\00061.png', help="path to IR image", required=False)
    # parser.add_argument("--v", default=r'E:\SIMpro\pro_infofusion\data\Vis\00061.png', help="path to IR image", required=False)
    # args = parser.parse_args()
    # VSMWLS(args.r, args.v)
    path_rgb = r"E:\SIMpro\pro_infofusion\fusionone\infra"
    path_ira = r"E:\SIMpro\pro_infofusion\fusionone\visrgb"
    csv_file = open('results_VSMWLS.csv', 'w', encoding='gbk', newline="")
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
        fus_img = VSMWLS(ira_id, rgb_id)
        # fus_img = ira_id
        cv2.imwrite('E://SIMpro//pro_infofusion//fusion_VSMWLS//results//' + '{}_fusion.png'.format(img_id), fus_img)
        visrgb = rgb_id
        infra = ira_id
        fusion_last = fus_img
        psnr = psnr_of_PSNR(visrgb, infra, fusion_last)
        psnr_vf = compare_psnr(visrgb, fusion_last)
        psnr_if = compare_psnr(infra, fusion_last)
        ssim2 = SSIM_GUAN(visrgb, infra, fusion_last)
        ssim3 = calculate_ssim_Gaussion(visrgb, fusion_last) + calculate_ssim_Gaussion(infra, fusion_last)
        print('psnr, psnr_vf, psnr_if, ssim2, ssim3 value: ', psnr, psnr_vf, psnr_if, ssim2, ssim3)

        info_csv = [img_id + '.png', psnr, ssim2, ssim3]
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
