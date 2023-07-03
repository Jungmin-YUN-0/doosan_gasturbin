import os
import numpy as np
import pickle
import glob 
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor

import segmentation_models_pytorch as smp
from utils.accuracy import accuracy_check


def main():

    
    img_path = '/HDD/tnwls/doosan/images/'
    mask_path = '/HDD/dataset/doosan/annotations/'
    data_list = glob.glob(img_path+'*.*')

    
    test_list = ['210415-AC1-1_m003_r1.png', '210415-AC8-1_m001_r1.png', '210415-AC8-1_m002_r1.png', \
    '210415-AC8-1_m003_r1.png', '210416-AC2-1_m001_r1.png', '210416-AC2-1_m002_r1.png', '210416-AC2-1_m003_r1.png']

    total_acc = 0
    for d in data_list:
        name = d.split('/')[-1]
        if name in test_list:
            doosan = Image.open(os.path.join(img_path, name))
            ori = Image.open(os.path.join(mask_path, name))

            doosan_np = np.asarray(doosan)
            ori_np = np.asarray(ori)

            total_acc += accuracy_check(ori_np, doosan_np)

    print('total_acc ', total_acc/len(test_list))
        

        

    # total_acc = inference(image_path, mask_path, model, save_dir)
    
    # print('TOTAL ACC!!', total_acc)
if __name__ == "__main__":
    main()
