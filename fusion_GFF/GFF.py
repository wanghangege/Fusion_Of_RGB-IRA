'''
    GFF是guided filter ng fusion(引导滤波图像融合)的缩写
    paper: 《Image Fusion with Guided Filtering》
    author(modify, not original): Benwu Wang
    date: 2022/11/11
'''
import csv
import os

import numpy as np
import cv2
import argparse
from fusionone.fusiononemmain import psnr_of_PSNR
from fusionone.fusiononemmain import calculate_ssim_Gaussion, SSIM_GUAN
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
R_G = 5
D_G = 5


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


def GFF_GRAY(img_r, img_v):
    img_r = img_r * 1. / 255
    img_v = img_v * 1. / 255
    img_r_blur = cv2.blur(img_r, (31, 31))
    img_v_blur = cv2.blur(img_v, (31, 31))
    img_r_detail = img_r.astype(np.float) - img_r_blur.astype(np.float)
    img_v_detail = img_v.astype(np.float) - img_v_blur.astype(np.float)
    img_r_lap = cv2.Laplacian(img_r.astype(np.float), -1, ksize=3)
    img_v_lap = cv2.Laplacian(img_v.astype(np.float), -1, ksize=3)
    win_size = 2 * R_G + 1
    s1 = cv2.GaussianBlur(np.abs(img_r_lap), (win_size, win_size), R_G)
    s2 = cv2.GaussianBlur(np.abs(img_v_lap), (win_size, win_size), R_G)
    p1 = np.zeros_like(img_r)
    p2 = np.zeros_like(img_r)
    p1[s1 > s2] = 1
    p2[s1 <= s2] = 1
    w1_b = guidedFilter(p1, img_r.astype(np.float), 45, 0.3)
    w2_b = guidedFilter(p2, img_v.astype(np.float), 45, 0.3)
    w1_d = guidedFilter(p1, img_r.astype(np.float), 7, 0.000001)
    w2_d = guidedFilter(p2, img_v.astype(np.float), 7, 0.000001)
    w1_b_w = w1_b / (w1_b + w2_b)
    w2_b_w = w2_b / (w1_b + w2_b)
    w1_d_w = w1_d / (w1_d + w2_d)
    w2_d_w = w2_d / (w1_d + w2_d)
    fused_b = w1_b_w * img_r_blur + w2_b_w * img_v_blur
    fused_d = w1_d_w * img_r_detail + w2_d_w * img_v_detail
    img_fused = fused_b + fused_d
    img_fused = cv2.normalize(img_fused, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.convertScaleAbs(img_fused)


def GFF_RGB(img_r, img_v):
    fused_img = np.ones_like(img_r)
    r_R = img_r[:, :, 2]
    v_R = img_v[:, :, 2]
    r_G = img_r[:, :, 1]
    v_G = img_v[:, :, 1]
    r_B = img_r[:, :, 0]
    v_B = img_v[:, :, 0]
    fused_R = GFF_GRAY(r_R, v_R)
    fused_G = GFF_GRAY(r_G, v_G)
    fused_B = GFF_GRAY(r_B, v_B)
    fused_img[:, :, 2] = fused_R
    fused_img[:, :, 1] = fused_G
    fused_img[:, :, 0] = fused_B
    return fused_img


def GFF(_rpath, _vpath):
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
            fused_img = GFF_GRAY(img_r, img_v)
        else:
            img_v_gray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
            fused_img = GFF_GRAY(img_r, img_v)
    else:
        if len(img_v.shape) < 3 or img_v.shape[-1] == 1:
            img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            fused_img = GFF_GRAY(img_r_gray, img_v)
        else:
            fused_img = GFF_RGB(img_r, img_v)
    return fused_img
    # cv2.imshow('fused image', fused_img)
    # cv2.imwrite("fused_image_gff.jpg", fused_img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # # parser.add_argument('-r', type=str, default='/home/wang/VIFB/TNO_Image_Fusion_Dataset/Athena_images/2_men_in_front_of_house/IR_meting003_g.bmp' ,help='input IR image path', required=False)
    # # parser.add_argument('-v', type=str, default= '/home/wang/VIFB/TNO_Image_Fusion_Dataset/Athena_images/2_men_in_front_of_house/VIS_meting003_r.bmp',help='input Visible image path', required=False)
    # parser.add_argument("--r", default=r'E:\SIMpro\pro_infofusion\data\Ir\00061.png', help="path to IR image",
    #                     required=False)
    # parser.add_argument("--v", default=r'E:\SIMpro\pro_infofusion\data\Vis\00061.png', help="path to IR image",
    #                     required=False)
    # args = parser.parse_args()
    # GFF(args.r, args.v)
    path_rgb = r"E:\SIMpro\pro_infofusion\fusionone\infra"
    path_ira = r"E:\SIMpro\pro_infofusion\fusionone\visrgb"
    csv_file =  open('results_GFF.csv', 'w', encoding='gbk', newline="")
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
        fus_img = GFF(ira_id, rgb_id)
        # fus_img = ira_id
        cv2.imwrite('E://SIMpro//pro_infofusion//fusion_GFF//results//' + '{}_fusion.png'.format(img_id), fus_img)
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

