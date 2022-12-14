import math
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import threading
import csv
import pywt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

class ImgFusion:
    def TIF_algo(self, p1, p2, median_blur_value=3, mean_blur_value=5):
        p1_b = cv2.blur(p1, (mean_blur_value, mean_blur_value))
        p1_b = p1_b.astype(np.float)
        p2_b = cv2.blur(p2, (mean_blur_value, mean_blur_value))
        p2_b = p2_b.astype(np.float)
        # cv2.imshow('picture after mean blur p1_b', p1_b)
        # cv2.imshow('picture after mean blur p2_b', p2_b)

        # p1_d = abs(p1.astype(np.float) - p1_b)
        # p2_d = abs(p2.astype(np.float) - p2_b)
        p1_d = p1.astype(np.float) - p1_b
        p2_d = p2.astype(np.float) - p2_b
        # cv2.imshow('detail layer p1', p1_d / 255.0)
        # cv2.imshow('detail layer p2', p2_d / 255.0)

        p1_after_medianblur = cv2.medianBlur(p1, median_blur_value)
        p2_after_medianblur = cv2.medianBlur(p2, median_blur_value)
        # cv2.imshow('picture after median blur p1_after_medianblur', p1_after_medianblur)
        # cv2.imshow('picture after median blur p2_after_medianblur', p2_after_medianblur)

        p1_after_medianblur = p1_after_medianblur.astype(np.float)
        p2_after_medianblur = p2_after_medianblur.astype(np.float)

        p1_subtract_from_median_mean = p1_after_medianblur - p1_b + 0.0001
        p2_subtract_from_median_mean = p2_after_medianblur - p2_b + 0.0001
        # cv2.imshow('subtract_from_median_mean  p1_subtract_from_median_mean', p1_subtract_from_median_mean/255.0)
        # cv2.imshow('subtract_from_median_mean  p2_subtract_from_median_mean', p2_subtract_from_median_mean/255.0)
        m1 = p1_subtract_from_median_mean[:, :, 0]
        m2 = p1_subtract_from_median_mean[:, :, 1]
        m3 = p1_subtract_from_median_mean[:, :, 2]
        res = m1 * m1 + m2 * m2 + m3 * m3
        delta1 = np.sqrt(res)
        # delta1 = res
        m1 = p2_subtract_from_median_mean[:, :, 0]
        m2 = p2_subtract_from_median_mean[:, :, 1]
        m3 = p2_subtract_from_median_mean[:, :, 2]
        res = m1 * m1 + m2 * m2 + m3 * m3

        delta2 = np.sqrt(res)

        # delta2 = abs(m1)

        delta_total = delta1 + delta2

        psi_1 = delta1 / delta_total
        psi_2 = delta2 / delta_total
        psi1 = np.zeros(p1.shape, dtype=np.float)
        psi2 = np.zeros(p2.shape, dtype=np.float)
        psi1[:, :, 0] = psi_1
        psi1[:, :, 1] = psi_1
        psi1[:, :, 2] = psi_1
        psi2[:, :, 0] = psi_2
        psi2[:, :, 1] = psi_2
        psi2[:, :, 2] = psi_2

        p_b = 0.5 * (p1_b + p2_b)
        # cv2.imshow('base pic1', p1_b / 255.0)
        # cv2.imshow('base pic2', p2_b / 255.0)
        # cv2.imshow('base pic', p_b / 255.0)

        p_d = psi1 * p1_d + psi2 * p2_d
        # cv2.imshow('detail layer plus', p_d / 255.0)
        # cv2.imshow('detail pic plus psi1 psi1 * p1_d', psi1 * p1_d)
        # cv2.imshow('detail pic plus psi1 psi2 * p2_d', psi2 * p2_d)
        p = p_b + p_d
        img = cv2.cvtColor(p.astype(np.float32) , cv2.COLOR_BGR2RGB)
        # cv2.imshow('final result', img.astype(np.float32)  / 255.0)
        # cv2.imwrite('./final_res.jpg', img)
        # cv2.waitKey(0)

        return img

    def strategy(self, arr_hvd1, arr_hvd2):
        k1 = 0.8
        k2 = 0.2
        arr_w1 = np.where(np.abs(arr_hvd1) > np.abs(arr_hvd2), k1, k2)
        arr_w2 = np.where(np.abs(arr_hvd1) < np.abs(arr_hvd2), k1, k2)
        return arr_w1, arr_w2

    def fusion(self, arr_visible, arr_infrared):
        it_h1 = arr_visible.shape[0]
        it_w1 = arr_visible.shape[1]
        it_h2 = arr_infrared.shape[0]
        it_w2 = arr_infrared.shape[1]
        if it_h1 % 2 != 0:
            it_h1 = it_h1 + 1
        if it_w1 % 2 != 0:
            it_w1 = it_w1 + 1
        if it_h2 % 2 != 0:
            it_h2 = it_h2 + 1
        if it_w2 % 2 != 0:
            it_w2 = it_w2 + 1
        arr_visible = cv2.resize(arr_visible, (it_w1, it_h1))
        arr_infrared = cv2.resize(arr_infrared, (it_w2, it_h2))

        it_level = 5

        arr_Gray1, arr_Gray2 = cv2.cvtColor(arr_visible, cv2.COLOR_BGR2GRAY), cv2.cvtColor(arr_infrared,
                                                                                           cv2.COLOR_BGR2GRAY)

        arr_Gray1 = arr_Gray1 * 1.0
        arr_Gray2 = arr_Gray2 * 1.0

        arr_visible = arr_visible * 1.0
        arr_infrared = arr_infrared * 1.0

        arr_decGray1 = pywt.wavedec2(arr_Gray1, 'sym4', level=it_level)
        arr_decGray2 = pywt.wavedec2(arr_Gray2, 'sym4', level=it_level)

        ls_decRed1 = pywt.wavedec2(arr_visible[:, :, 0], 'sym4', level=it_level)
        ls_decGreen1 = pywt.wavedec2(arr_visible[:, :, 1], 'sym4', level=it_level)
        ls_decBlue1 = pywt.wavedec2(arr_visible[:, :, 2], 'sym4', level=it_level)
        ls_recRed = []
        ls_recGreen = []
        ls_recBlue = []

        for it_i, (arr_gray1, arr_gray2, arr_red1, arr_green1, arr_blue1) in enumerate(
                zip(arr_decGray1, arr_decGray2, ls_decRed1, ls_decGreen1, ls_decBlue1)):
            if it_i == 0:
                fl_w1 = 0.5
                fl_w2 = 0.5
                us_recRed = fl_w1 * arr_red1 + fl_w2 * arr_gray2
                us_recGreen = fl_w1 * arr_green1 + fl_w2 * arr_gray2
                us_recBlue = fl_w1 * arr_blue1 + fl_w2 * arr_gray2
            else:
                us_recRed = []
                us_recGreen = []
                us_recBlue = []
                for arr_grayHVD1, arr_grayHVD2, arr_redHVD1, arr_greenHVD1, arr_blueHVD1, in zip(arr_gray1, arr_gray2,
                                                                                                 arr_red1, arr_green1,
                                                                                                 arr_blue1):
                    arr_w1, arr_w2 = self.strategy(arr_grayHVD1, arr_grayHVD2)
                    arr_recRed = arr_w1 * arr_redHVD1 + arr_w2 * arr_grayHVD2
                    arr_recGreen = arr_w1 * arr_greenHVD1 + arr_w2 * arr_grayHVD2
                    arr_recBlue = arr_w1 * arr_blueHVD1 + arr_w2 * arr_grayHVD2

                    us_recRed.append(arr_recRed)
                    us_recGreen.append(arr_recGreen)
                    us_recBlue.append(arr_recBlue)

            ls_recRed.append(us_recRed)
            ls_recGreen.append(us_recGreen)
            ls_recBlue.append(us_recBlue)

        arr_rec = np.zeros(arr_visible.shape)
        arr_rec[:, :, 0] = pywt.waverec2(ls_recRed, 'sym4')
        arr_rec[:, :, 1] = pywt.waverec2(ls_recGreen, 'sym4')
        arr_rec[:, :, 2] = pywt.waverec2(ls_recBlue, 'sym4')

        # img = cv2.cvtColor(p.astype(np.float32) , cv2.COLOR_BGR2RGB)
        # cv2.imshow('final result', arr_rec.astype(np.float32)  / 255.0)
        # cv2.imwrite('./final_res.jpg', arr_rec)
        # cv2.waitKey(0)

        return arr_rec

def mse_of_PNSR(visrgb, infra, fusion_last):
    # size_H = visrgb.shape[0]
    # size_W = visrgb.shape[1]
    # size_Channels = visrgb.shape[2]
    return 0.5 * np.mean(
        (visrgb.astype(np.float64) / 1.0 - fusion_last.astype(np.float64) / 1.0) ** 2 + \
        (infra.astype(np.float64) / 1.0 - fusion_last.astype(np.float64) / 1.0 ) ** 2
            )

def psnr_of_PSNR(visrgb, infra, fusion_last, n_bite=8):
    # visrgb = cv2.normalize(visrgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    # infra = cv2.normalize(infra, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    # fusion_last = cv2.normalize(fusion_last, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    # visrgb = visrgb.astype(np.uint8)
    # infra = infra.astype(np.uint8)
    # fusion_last = fusion_last.astype(np.uint8)
    visrgb = visrgb.astype(np.float64)
    infra = infra.astype(np.float64)
    fusion_last = fusion_last.astype(np.float64)
    mse_value = mse_of_PNSR(visrgb, infra, fusion_last)
    if mse_value == 0.:
        return np.inf

    return 10 * np.log10((2**n_bite - 1) ** 2 / mse_value)
    # return 20 * math.log10(1.0 / math.sqrt(mse_value))


def VI_ADD_IR_GRAY(visrgb, infra, fusion_last):
    visrgb = cv2.normalize(visrgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    # visrgb = visrgb.astype(np.uint8)
    infra = cv2.normalize(infra, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    # infra = infra.astype(np.uint8)
    fusion_last = cv2.normalize(fusion_last, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    # fusion_last = fusion_last.astype(np.uint8)
    #??????
    mean_vis = np.mean(visrgb)
    mean_ira = np.mean(infra)
    mean_fus = np.mean(fusion_last)
    #??????var
    var_vis = np.var(visrgb)
    var_ira = np.var(infra)
    var_fus = np.var(fusion_last)
    # var_vis = np.mean((visrgb - mean_vis) ** 2)  * 255
    # var_ira = np.mean((infra - mean_ira) ** 2)  * 255
    # var_fus = np.mean((fusion_last - mean_fus) ** 2)  * 255
    #?????????cov
    cov_vis_fus = np.mean(np.cov(visrgb, fusion_last))
    cov_ira_fus = np.mean(np.cov(infra, fusion_last))
    # cov_vis_fus = np.mean((visrgb - mean_vis) * (fusion_last - mean_fus))
    # cov_ira_fus = np.mean((infra - mean_ira) * (fusion_last - mean_fus))
    #?????????
    std_vis = np.std(visrgb)
    std_ira = np.std(infra)
    std_fus = np.std(fusion_last)
    #????????????
    K1 = 0.01
    K2 = 0.03
    L = 255
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2
    c3 = c2 / 2
    ##??????VI
    l_vis_fus = (2 * mean_vis * mean_fus + c1) / (mean_vis ** 2 + mean_fus ** 2 + c1)
    c_vis_fus = (2 * std_vis * std_fus + c2) / (std_vis ** 2 + std_fus ** 2 + c2)
    s_vis_fus = (cov_vis_fus + c3) / (std_vis * std_fus + c3)

    ##??????IR
    l_ira_fus = (2 * mean_ira * mean_fus + c1) / (mean_ira ** 2 + mean_fus ** 2 + c1)
    c_ira_fus = (2 * std_ira * std_fus + c2) / (std_ira ** 2 + std_fus ** 2 + c2)
    s_ira_fus = (cov_ira_fus + c3) / (std_ira * std_fus + c3)
    VI = l_vis_fus * c_vis_fus * s_vis_fus
    IR = l_ira_fus * c_ira_fus * s_ira_fus
    # VI = l_vis_fus * c_vis_fus
    # IR = l_ira_fus * c_ira_fus
    return VI + IR

def ssim(visrgb, infra, fusion_last):
    # if visrgb.shape[2] == 3 and infra.shape[2] == 3 and fusion_last.shape[2] == 3:
    #     print('??????????????????')
    ssim_in = 0
    for i in range(0, 3):
        ssim_temp = VI_ADD_IR_GRAY(visrgb[:, :, i], infra[:, :, i], fusion_last[:, :, i])
        ssim_in += ssim_temp

    return ssim_in / 3

def SSIM_GUAN(visrgb, infra, fusion_last):
    visrgb = cv2.normalize(visrgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    infra = cv2.normalize(infra, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    fusion_last = cv2.normalize(fusion_last, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    # if visrgb.shape[2] == 3 and infra.shape[2] == 3 and fusion_last.shape[2] == 3:
    #     print('??????????????????')
    # ssim_in = 0
    # for i in range(0, 3):
    #     VI = compare_ssim(visrgb, fusion_last, data_range=255, multichannel=True)
    #     IR = compare_ssim(infra, fusion_last, data_range=255, multichannel=True)
    #     ssim_in = ssim_in + VI + IR
    VI = compare_ssim(visrgb, fusion_last, data_range=255, multichannel=True)
    IR = compare_ssim(infra, fusion_last, data_range=255, multichannel=True)
    ssim_in =  VI + IR
    return ssim_in


def ssim_Gaussian(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim_Gaussion(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    img2 = cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    img1 = np.array(img1)
    img2 = np.array(img2)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim_Gaussian(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_Gaussian(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim_Gaussian(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')




if __name__ =='__main__':
    # img_rgb=cv2.imread(r'E:\SIMpro\pro_infofusion\fusionone\visrgb\00061.png')
    # img_t=cv2.imread(r'E:\SIMpro\pro_infofusion\fusionone\infra\00061.png')

    path_rgb = r"E:\SIMpro\pro_infofusion\fusionone\infra"
    path_ira = r"E:\SIMpro\pro_infofusion\fusionone\visrgb"

    f=ImgFusion()
    csv_file =  open('results.csv', 'w', encoding='gbk', newline="")
    csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(["image name", "pnsr value", "official ssim value", "my ssim value"])
    csv_writer.writerow(["????????????", "PNSR???", "SSIM???", "SSIM??????????????????"])


    for img_id in os.listdir(path_rgb):
        img_id = img_id.split('.')[0]
        print(img_id)
        rgb_id = cv2.imread(os.path.join(path_rgb, img_id + '.png'))
        ira_id = cv2.imread(os.path.join(path_ira, img_id + '.png'))
        ##??????????????????
        # fus_img = f.TIF_algo(rgb_id, ira_id)
        fus_img = f.fusion(rgb_id, ira_id)
        fus_img = np.where(fus_img > 255.0, 255.0, fus_img)
        fus_img = np.where(fus_img < 0.0, 0.0, fus_img)
        plt.figure('???????????????')
        plt.subplot(131)
        plt.imshow(rgb_id.astype(np.uint8))
        plt.subplot(132)
        plt.imshow(ira_id.astype(np.uint8))
        plt.subplot(133)
        plt.imshow(fus_img.astype(np.uint8))
        # cv2.imwrite('./results/fusion_img/%d_rgb.png' % int(img_id), rgb_id)
        # cv2.imwrite('./results/fusion_img/%d_ira_id.png' % int(img_id), ira_id)
        # cv2.imwrite('./results/fusion_img/%d_fusion.png' % int(img_id), fus_img)
        psnr = psnr_of_PSNR(rgb_id, ira_id, fus_img)
        print(psnr)
        # plt.show()
        # ssim1 = ssim(rgb_id, ira_id, fus_img)
        # print('?????????ssim???', ssim1)
        ssim2 = SSIM_GUAN(rgb_id, ira_id, fus_img)
        # print('??????ssim??????', ssim2)
        ssim3 = calculate_ssim_Gaussion(rgb_id, fus_img) + calculate_ssim_Gaussion(ira_id, fus_img)
        # print('?????????ssim??????', ssim3)

        info_csv = [str(img_id), psnr, ssim2, ssim3]
        csv_writer.writerow(info_csv)
    csv_file.close()

