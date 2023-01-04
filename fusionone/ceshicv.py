import cv2
import numpy as np
import math

def VI_ADD_IR_GRAY(visrgb, infra, fusion_last):

    visrgb = cv2.normalize(visrgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    infra = cv2.normalize(infra, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    fusion_last = cv2.normalize(fusion_last, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255

    #均值
    mean_vis = np.mean(visrgb)
    mean_ira = np.mean(infra)
    mean_fus = np.mean(fusion_last)
    #方差var
    var_vis = np.mean((visrgb - mean_vis) ** 2)
    var_ira = np.mean((infra - mean_ira) ** 2)
    var_fus = np.mean((fusion_last - mean_fus) ** 2)
    #协方差cov
    cov_vis_fus = np.mean((visrgb - mean_vis) * (fusion_last - mean_fus))
    cov_ira_fus = np.mean((infra - mean_ira) * (fusion_last - mean_fus))

    #公共参数
    K1 = 0.01
    K2 = 0.03
    L = 255
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2
    c3 = c2 / 2

    ##计算VI
    l_vis_fus = (2 * mean_vis * mean_fus + c1) / (mean_vis ** 2 + mean_fus ** 2 + c1)
    c_vis_fus = (2 * math.sqrt(var_vis) * math.sqrt(var_fus) + c2) / (var_vis + var_fus + c2)
    s_vis_fus = (cov_vis_fus + c3) / (math.sqrt(var_vis) * math.sqrt(var_fus) + c3)

    ##计算IR
    l_ira_fus = (2 * mean_ira * mean_fus + c1) / (mean_ira ** 2 + mean_fus ** 2 + c1)
    c_ira_fus = (2 * math.sqrt(var_ira) * math.sqrt(var_fus) + c2) / (var_ira + var_fus + c2)
    s_ira_fus = (cov_ira_fus + c3) / (math.sqrt(var_ira) * math.sqrt(var_fus) + c3)

    VI = l_vis_fus * c_vis_fus * s_vis_fus
    IR = l_ira_fus * c_ira_fus * s_ira_fus
    return VI + IR


img_r = cv2.imread(r"E:\SIMpro\pro_infofusion\fusionone\visrgb\00061.png", cv2.IMREAD_GRAYSCALE)
img_i = cv2.imread(r"E:\SIMpro\pro_infofusion\fusionone\infra\00061.png", cv2.IMREAD_GRAYSCALE)
img_f = cv2.imread(r"E:\SIMpro\pro_infofusion\fusionone\final_res.jpg", cv2.IMREAD_GRAYSCALE)
det = VI_ADD_IR_GRAY(img_r, img_i, img_f)
print(det)