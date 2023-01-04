"""
    网络部分来自于PIAFusion，本节只包含推理部分
    源代码：https://github.com/Linfeng-Tang/PIAFusion
    原论文：https://www.sciencedirect.com/science/article/abs/pii/S156625352200032X
    改动者：WANG Benwu   单位：SEU
    目的：用于融合可见光与红外图像的融合，并进行PNSR和SSIM指标评价
"""
import argparse
import os
import random
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from models.fusion_model import PIAFusion
from PIL import Image
from models.metric_fun import psnr_of_PSNR, SSIM_GUAN, calculate_ssim_Gaussion, ssim
import csv
import cv2

to_tensor = transforms.Compose([transforms.ToTensor()])


class MSRS_data(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)  # 获得TNO数据集的子目录
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'infra':
                self.inf_path = temp_path  # 获得红外路径
            elif sub_dir == 'visrgb':
                self.vis_path = temp_path  # 获得可见光路径

        self.name_list = os.listdir(self.inf_path)  # 获得子目录下的图片的名称
        self.transform = transform
    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称

        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')  # 获取红外图像
        vis_image = Image.open(os.path.join(self.vis_path, name))
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name
    def __len__(self):
        return len(self.name_list)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args_cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)

def YCrCb2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式

    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out
def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式

    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr

if __name__ == '__main__':
    args_cuda = True
    init_seeds(int(0))
    save_path = 'data/fusimg'
    dataset_path = 'data'
    fusion_pretrained = 'pretrained/fusion_model_epoch_29.pth'

    test_dataset = MSRS_data(dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 如果是融合网络
    model = PIAFusion()
    model = model.cuda()
    model.load_state_dict(torch.load(fusion_pretrained))
    model.eval()
    test_tqdm = tqdm(test_loader, total=len(test_loader))
    csv_file =  open('results.csv', 'w', encoding='gbk', newline="")
    csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(["image name", "pnsr value", "official ssim value", "my ssim value"])
    csv_writer.writerow(["ImageName", "PNSR", "SSIM_A", "SSIM_B", "SSIM_C"])

    psnr_v = []
    ssim_v1 = []
    ssim_v2 = []
    ssim_v3 = []
    with torch.no_grad():
        for _, vis_y_image, cb, cr, inf_image, name in test_tqdm:
            vis_y_image = vis_y_image.cuda()
            cb = cb.cuda()
            cr = cr.cuda()
            inf_image = inf_image.cuda()

            # 测试转为Ycbcr的数据再转换回来的输出效果，结果与原图一样，说明这两个函数是没有问题的。
            # t = YCbCr2RGB2(vis_y_image[0], cb[0], cr[0])
            # transforms.ToPILImage()(t).save(name[0])

            fused_image = model(vis_y_image, inf_image)
            fused_image = clamp(fused_image)

            rgb_fused_image = YCrCb2RGB(fused_image[0], cb[0], cr[0])
            rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
            rgb_fused_image.save(f'{save_path}/{name[0]}')

            rgb_id = cv2.imread(os.path.join(dataset_path + '/visrgb', name[0]))
            ira_id = cv2.imread(os.path.join(dataset_path + '/infra', name[0]))
            fus_img = cv2.cvtColor(np.asarray(rgb_fused_image), cv2.COLOR_RGB2BGR)
            #开始评价指标：
            psnr = psnr_of_PSNR(rgb_id, ira_id, fus_img)
            ssim1 = ssim(rgb_id, ira_id, fus_img)
            ssim2 = SSIM_GUAN(rgb_id, ira_id, fus_img)
            ssim3 = calculate_ssim_Gaussion(rgb_id, fus_img) + calculate_ssim_Gaussion(ira_id, fus_img)
            print('自定义psnr：', psnr)
            print('自定义ssim：', ssim1)
            print('官方的ssim：', ssim2)
            print('第三方ssim：', ssim3)

            psnr_v.append(psnr)
            ssim_v1.append(ssim1)
            ssim_v2.append(ssim2)
            ssim_v3.append(ssim3)
            info_csv = [name[0], psnr, ssim2, ssim3, ssim1]
            csv_writer.writerow(info_csv)
    csv_file.close()
    print('psnr平均值： ', np.mean(psnr))
    print('ssim1平均值： ', np.mean(ssim_v1))
    print('ssim2平均值： ', np.mean(ssim_v2))
    print('ssim3平均值： ', np.mean(ssim_v3))
