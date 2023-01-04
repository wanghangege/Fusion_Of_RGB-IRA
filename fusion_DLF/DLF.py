'''
    DLF是Deep Learning Framework(深度学习框架)的缩写
    paper: 《Infrared and Visible Image Fusion using a Deep Learning Framework》
    author(modify, not original): Benwu Wang
    date: 2022/11/11
'''
import csv
import os

import numpy as np
from sporco.signal import tikhonov_filter
import scipy
import torch
from torchvision.models.vgg import vgg19
import cv2
import argparse
from fusionone.fusiononemmain import psnr_of_PSNR
from fusionone.fusiononemmain import calculate_ssim_Gaussion, SSIM_GUAN
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def lowpass(s, lda, npad):
    return tikhonov_filter(s, lda, npad)


def c3(s):
    if s.ndim == 2:
        s3 = np.dstack([s, s, s])
    else:
        s3 = s
    return np.rollaxis(s3, 2, 0)[None, :, :, :]


def l1_features(out):
    h, w, d = out.shape
    A_temp = np.zeros((h + 2, w + 2))

    l1_norm = np.sum(np.abs(out), axis=2)
    A_temp[1:h + 1, 1:w + 1] = l1_norm
    return A_temp


def fusion_strategy(feat_a, feat_b, source_a, source_b, unit):
    m, n = feat_a.shape
    m1, n1 = source_a.shape[:2]
    weight_ave_temp1 = np.zeros((m1, n1))
    weight_ave_temp2 = np.zeros((m1, n1))

    for i in range(1, m):
        for j in range(1, n):
            A1 = feat_a[i - 1:i + 1, j - 1:j + 1].sum() / 9
            A2 = feat_b[i - 1:i + 1, j - 1:j + 1].sum() / 9

            weight_ave_temp1[(i - 2) * unit + 1:(i - 1) * unit + 1, (j - 2) * unit + 1:(j - 1) * unit + 1] = A1 / (
                        A1 + A2)
            weight_ave_temp2[(i - 2) * unit + 1:(i - 1) * unit + 1, (j - 2) * unit + 1:(j - 1) * unit + 1] = A2 / (
                        A1 + A2)

    if source_a.ndim == 3:
        weight_ave_temp1 = weight_ave_temp1[:, :, None]
    source_a_fuse = source_a * weight_ave_temp1
    if source_b.ndim == 3:
        weight_ave_temp2 = weight_ave_temp2[:, :, None]
    source_b_fuse = source_b * weight_ave_temp2

    if source_a.ndim == 3 or source_b.ndim == 3:
        gen = np.atleast_3d(source_a_fuse) + np.atleast_3d(source_b_fuse)
    else:
        gen = source_a_fuse + source_b_fuse

    return gen


def get_activation(model, layer_numbers, input_image):
    outs = []
    out = input_image
    for i in range(max(layer_numbers) + 1):
        with torch.no_grad():
            out = model.features[i](out)
        if i in layer_numbers:
            outs.append(np.rollaxis(out.detach().cpu().numpy()[0], 0, 3))
    return outs


def DLF(vis, ir, model=None):
    # vis = cv2.imread(vis)
    # ir = cv2.imread(ir)
    npad = 16
    lda = 5
    vis_low, vis_high = lowpass(vis.astype(np.float32) / 255, lda, npad)
    ir_low, ir_high = lowpass(ir.astype(np.float32) / 255, lda, npad)

    if model is None:
        model = vgg19(True)
    model.cuda().eval()
    relus = [2, 7, 12, 21]
    unit_relus = [1, 2, 4, 8]

    vis_in = torch.from_numpy(c3(vis_high)).cuda()
    ir_in = torch.from_numpy(c3(ir_high)).cuda()

    relus_vis = get_activation(model, relus, vis_in)
    relus_ir = get_activation(model, relus, ir_in)

    vis_feats = [l1_features(out) for out in relus_vis]
    ir_feats = [l1_features(out) for out in relus_ir]

    saliencies = []
    saliency_max = None
    for idx in range(len(relus)):
        saliency_current = fusion_strategy(vis_feats[idx], ir_feats[idx], vis_high, ir_high, unit_relus[idx])
        saliencies.append(saliency_current)

        if saliency_max is None:
            saliency_max = saliency_current
        else:
            saliency_max = np.maximum(saliency_max, saliency_current)

    if vis_low.ndim == 3 or ir_low.ndim == 3:
        low_fused = np.atleast_3d(vis_low) + np.atleast_3d(ir_low)
    else:
        low_fused = vis_low + ir_low
    low_fused = low_fused / 2
    high_fused = saliency_max
    fused_img = low_fused + high_fused
    fused_img = cv2.normalize(fused_img, None, 255., 0., cv2.NORM_MINMAX)
    fused_img = cv2.convertScaleAbs(fused_img)
    return fused_img
    # cv2.imshow('fused image', fused_img)
    # cv2.waitKey(0)
    # fused_img = cv2.normalize(fused_img, None, 255., 0., cv2.NORM_MINMAX)
    # fused_img = cv2.convertScaleAbs(fused_img)
    # cv2.imwrite("fused_image_dlf.jpg", fused_img)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--IR",  default='/home/wang/VIFB/IV_images/IR1.png', help="path to IR image", required=False)
    # # parser.add_argument("--VIS",  default='/home/wang/VIFB/IV_images/VIS1.png', help="path to IR image", required=False)
    # parser.add_argument("--IR", default=r'E:\SIMpro\pro_infofusion\data\Ir\00061.png', help="path to IR image",
    #                     required=False)
    # parser.add_argument("--VIS", default=r'E:\SIMpro\pro_infofusion\data\Vis\00061.png', help="path to IR image",
    #                     required=False)
    # a = parser.parse_args()
    # DLF(a.IR, a.VIS)
    path_rgb = r"E:\SIMpro\pro_infofusion\fusionone\infra"
    path_ira = r"E:\SIMpro\pro_infofusion\fusionone\visrgb"
    csv_file =  open('results_DLF.csv', 'w', encoding='gbk', newline="")
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
        fus_img = DLF(ira_id, rgb_id)
        # fus_img = ira_id
        cv2.imwrite('E://SIMpro//pro_infofusion//fusion_DLF//results//' + '{}_fusion.png'.format(img_id), fus_img)
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
