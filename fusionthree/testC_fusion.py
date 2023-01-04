'''
    网络部分：DIFNet
    paper：《DIFNet: Boosting Visual Information Flow for Image Captioning》
    code author: Benwu Wang
    date: 2022/11/11
'''
import csv
import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm
from PIL import Image
from time import time
import math

from model.network import DIFNet
from model.args import args

from fusionone.fusiononemmain import psnr_of_PSNR
from fusionone.fusiononemmain import calculate_ssim_Gaussion, SSIM_GUAN
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

is_cuda = torch.cuda.is_available()


class MyTestDataset(Dataset):
    def __init__(self, img1_path_file, img2_path_file):
        # f1 = open(img1_path_file, 'r')
        # img1_list = f1.read().splitlines()
        # f1.close()
        # f2 = open(img2_path_file, 'r')
        # img2_list = f2.read().splitlines()
        # f2.close()
        img1_list = []
        img2_list = []
        for img in os.listdir(img1_path_file):
            vis_path = os.path.join(img1_path_file, img)
            ira_path = os.path.join(img2_path_file, img)
            img1_list.append(vis_path)
            img2_list.append(ira_path)

        self.img1_list = img1_list
        self.img2_list = img2_list

    def __getitem__(self, index):
        img1 = Image.open(self.img1_list[index]).convert('RGB')
        img2 = Image.open(self.img2_list[index]).convert('RGB')

        custom_transform_rgb = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                   transforms.Resize((args.HEIGHT, args.WIDTH)),
                                                   transforms.ToTensor()])
        custom_transform_gray = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                    transforms.Resize((args.HEIGHT, args.WIDTH)),
                                                    transforms.ToTensor()])

        img1 = custom_transform_rgb(img1)
        img2 = custom_transform_gray(img2)
        img_name = self.img1_list[index]
        img_name = img_name.split('\\')[1]

        return img_name, img1, img2

    def __len__(self):
        return len(self.img1_list)


if __name__ == '__main__':

    model = DIFNet()

    checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # img1_path_file = args.test_visible
    # img2_path_file = args.test_lwir
    path_visrgb = 'data/visrgb'
    path_infra  = 'data/infra'

    testloader = DataLoader(MyTestDataset(path_visrgb, path_infra), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)

    if is_cuda:
        model.cuda()
    psnr_v = []
    ssim_v2 = []
    ssim_v3 = []
    psnr_vf_v, psnr_if_v = [], []

    # 写入文档
    csv_file = open('results.csv', 'w', encoding='gbk', newline="")
    csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(["image name", "pnsr value", "official ssim value", "my ssim value"])
    csv_writer.writerow(["ImageName", "PNSR", "SSIM_A", "SSIM_B", "SSIM_C"])

    with torch.no_grad():
        for i, (img_name, img_1, img_2) in enumerate(tqdm(testloader)):

            img_cat = torch.cat([img_1, img_2], dim=1)
            if is_cuda and args.gpu is not None:
                img_1 = img_1.cuda()
                img_2 = img_2.cuda()
                img_cat = img_cat.cuda()

            fusion = model(img_cat)

            if is_cuda:
                fusion = fusion.cpu()
            else:
                pass
            visrgb = img_1.cpu().numpy()[0, 0, :, :]
            infra = img_2.cpu().numpy()[0, 0, :, :]
            fusion_last = fusion.cpu().detach().numpy()[0, 0, :, :]
            psnr = psnr_of_PSNR(visrgb, infra, fusion_last)
            psnr_vf = compare_psnr(visrgb, fusion_last)
            psnr_if = compare_psnr(infra, fusion_last)
            ssim2 = SSIM_GUAN(visrgb, infra, fusion_last)
            ssim3 = calculate_ssim_Gaussion(visrgb, fusion_last) + calculate_ssim_Gaussion(infra, fusion_last)
            print('psnr, psnr_vf, psnr_if, ssim2, ssim3 value: ', psnr, psnr_vf, psnr_if, ssim2, ssim3)

            info_csv = [img_name[0], psnr, ssim2, ssim3]
            csv_writer.writerow(info_csv)

            psnr_v.append(psnr)
            ssim_v2.append(ssim2)
            ssim_v3.append(ssim3)
            psnr_vf_v.append(psnr_vf)
            psnr_if_v.append(psnr_if)
            # save_image(img_1, args.test_save_dir + '{}_visible.png'.format(i))
            # save_image(img_2, args.test_save_dir + '{}_ir.png'.format(i))
            save_image(fusion, args.test_save_dir + '{}_fusion.png'.format(i))
            # save_image((img_1 + img_2) / 2, args.test_save_dir + '{}_add.png'.format(i))

        print('Finished testing')
        print('The evaluted vale:')
        print('psnr_v : ', np.mean(psnr_v))
        print('ssim_v2: ', np.mean(ssim_v2))
        print('ssim_v3: ', np.mean(ssim_v3))
        print('psnr_vf_v: ', np.mean(psnr_vf_v))
        print('psnr_if_v: ', np.mean(psnr_if_v))
        csv_file.close()


