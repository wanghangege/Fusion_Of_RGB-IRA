'''
    MGFF是Multi-scale Guided Image and Video Fusion(多尺度引导图像与视频融合)的缩写
    paper: 《Multi-scale Guided Image and Video Fusion: A Fast and Efficient Approach》
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
ITERATION = 4
RADIUS = 9
EPS = 10 ** 3 / (256 ** 2)


def guidedFilter(img_i, img_p, r, eps):
    wsize = int(2 * r) + 1
    meanI = cv2.boxFilter(img_i, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    meanP = cv2.boxFilter(img_p, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    corr_I = cv2.boxFilter(img_i * img_i, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    corrIP = cv2.boxFilter(img_i * img_p, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    varI = corr_I - meanI * meanI
    covIP = corrIP - meanI * meanP
    a = covIP / (varI + eps)
    b = meanP - a * meanI
    meanA = cv2.boxFilter(a, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    meanB = cv2.boxFilter(b, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    q = meanA * img_i + meanB
    return q


def MGFF_GRAY(img_r, img_v):
    img_r_base_pre = img_r * 1. / 255
    img_v_base_pre = img_v * 1. / 255
    img_r_detail = []
    img_v_detail = []
    for i in range(ITERATION):
        img_r_base_cur = guidedFilter(img_v_base_pre, img_r_base_pre, RADIUS, EPS)
        img_v_base_cur = guidedFilter(img_r_base_pre, img_v_base_pre, RADIUS, EPS)
        img_r_detail.append(img_r_base_pre - img_r_base_cur)
        img_v_detail.append(img_v_base_pre - img_v_base_cur)
        img_r_base_pre = img_r_base_cur
        img_v_base_pre = img_v_base_cur
    img_base_fused = (img_r_base_pre + img_v_base_pre) / 2
    fused_img = img_base_fused
    for i in range(ITERATION - 1, -1, -1):
        weights = np.abs(img_r_detail[i]) / (np.abs(img_r_detail[i]) + np.abs(img_v_detail[i]))
        fused_img += weights * img_r_detail[i] + (1 - weights) * img_v_detail[i]
    fused_img = cv2.normalize(fused_img, None, 0., 255., cv2.NORM_MINMAX)
    fused_img = cv2.convertScaleAbs(fused_img)
    return fused_img


def MGFF_RGB(img_r, img_v):
    r_R = img_r[:, :, 2]
    r_G = img_r[:, :, 1]
    r_B = img_r[:, :, 0]
    v_R = img_v[:, :, 2]
    v_G = img_v[:, :, 1]
    v_B = img_v[:, :, 0]
    fused_R = MGFF_GRAY(r_R, v_R)
    fused_G = MGFF_GRAY(r_G, v_G)
    fused_B = MGFF_GRAY(r_B, v_B)
    fused_img = np.stack((fused_B, fused_G, fused_R), axis=-1)
    return fused_img


def MGFF(r_path, v_path):
    # img_r = cv2.imread(r_path)
    # img_v = cv2.imread(v_path)
    img_r = r_path
    img_v = v_path
    if not isinstance(img_r, np.ndarray):
        print("img_r is not an image")
        return
    if not isinstance(img_v, np.ndarray):
        print("img_v is not an image")
        return
    fused_img = None
    if len(img_r.shape) == 2 or img_r.shape[-1] == 1:
        if img_r.shape[-1] == 3:
            img_v = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
        fused_img = MGFF_GRAY(img_r, img_v)
    else:
        if img_r.shape[-1] == 3:
            fused_img = MGFF_RGB(img_r, img_v)
        else:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            fused_img = MGFF_GRAY(img_r, img_v)
    return fused_img
    # cv2.imshow("fused image", fused_img)
    # cv2.imwrite("fused_image_mgff.jpg", fused_img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--IR",  default='/home/wang/VIFB/IV_images/IR6.png', help="path to IR image", required=False)
    # # parser.add_argument("--VIS",  default='/home/wang/VIFB/IV_images/VIS6.png', help="path to IR image", required=False)
    # parser.add_argument("--IR", default=r'E:\SIMpro\pro_infofusion\data\Ir\00061.png', help="path to IR image",
    #                     required=False)
    # parser.add_argument("--VIS", default=r'E:\SIMpro\pro_infofusion\data\Vis\00061.png', help="path to IR image",
    #                     required=False)
    #
    # a = parser.parse_args()
    # MGFF(a.IR, a.VIS)
    path_rgb = r"E:\SIMpro\pro_infofusion\fusionone\infra"
    path_ira = r"E:\SIMpro\pro_infofusion\fusionone\visrgb"
    csv_file =  open('results_MGFF.csv', 'w', encoding='gbk', newline="")
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
        fus_img = MGFF(ira_id, rgb_id)
        # fus_img = ira_id
        cv2.imwrite('E://SIMpro//pro_infofusion//fusion_MGFF//results//' + '{}_fusion.png'.format(img_id), fus_img)
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
