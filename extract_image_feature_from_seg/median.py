import os
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor

import segmentation_models_pytorch as smp
from utils.accuracy import accuracy_check
import matplotlib.pyplot as plt


def main():

    data_root_path = '/HDD/dataset/doosan/eta/'
    image_path = '/HDD/dataset/doosan/eta/origin_img'
    mask_path = '/HDD/dataset/doosan/eta/origin_msk'
    model_path = '/HDD/jungmin/doosan/history/deep/eta_best/saved_models'

    with open(os.path.join(data_root_path, 'images.txt'), 'rb') as f:
        data_list = pickle.load(f)

    save_dir = os.path.join('/'.join(model_path.split('/')[:-1]), 'inference_images/')
    # save_dir = os.path.join(image_path, 'inference_images/')
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = smp.DeepLabV3('resnet18', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)
    # model = smp.Unet('vgg16_bn', in_channels=1, classes=1)
    # model = smp.DeepLabV3Plus('timm-regnety_320',  encoder_depth=5, encoder_output_stride=16, decoder_channels=512, in_channels=1) #relu encoder_depth=4,
    # model = smp.DeepLabV3('resnet34', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)

    model.load_state_dict(torch.load(os.path.join(model_path, 'model_best.pt')))

    model.cuda()

    # Train
    total_acc = 0
    toTensor = ToTensor() 
    minmax_scaler = MinMaxScaler()
    model.eval()

    for data in data_list:
        data = data.split(' ')[0]
        img = Image.open(os.path.join(image_path, data))
        msk = Image.open(os.path.join(mask_path, data))
        
        img = toTensor(img).cuda().unsqueeze(0)
        msk = toTensor(msk).cuda().unsqueeze(0)

        with torch.no_grad():
            output = model(img)

        # print(output.shape)
        result = minmax_scaler.fit_transform(output[0][0].cpu())
        result[result < 0.5] = 0
        result[result >= 0.5] = 255

        mask_ori = minmax_scaler.fit_transform(msk[0][0].cpu())
        mask_ori[mask_ori < 0.5] = 0
        mask_ori[mask_ori >= 0.5] = 255

        palette = [0,0,0, 255,255,255]
        out = Image.fromarray(result.astype(np.uint8), mode='P')
        out.putpalette(palette)

        export_name = str(data)
        out.save(save_dir + export_name)
        acc = accuracy_check(mask_ori, out)
        total_acc += acc

        # fig = plt.figure()
        # rows = 1
        # cols = 3

        # ax1 = fig.add_subplot(rows, cols, 1)
        # ax1.imshow(img[0][0].cpu().numpy(), 'gray')
        # ax1.set_title('original image')
        # ax1.axis("off")

        # ax2 = fig.add_subplot(rows, cols, 2)
        # ax2.imshow(result.astype(np.uint8), 'gray')
        # ax2.set_title('prediction')
        # ax2.axis("off")

        # ax3 = fig.add_subplot(rows, cols, 3)
        # ax3.imshow(mask_ori, 'gray')
        # ax3.set_title('ground truth')
        # ax3.axis("off")

        # export_name_fig = export_name[:-4] +'_fig.png'
        # plt.savefig(save_dir + export_name_fig)
    
    print('total_acc : ', total_acc/len(data_list))
        

        

    # total_acc = inference(image_path, mask_path, model, save_dir)
    
    # print('TOTAL ACC!!', total_acc)
if __name__ == "__main__":
    main()
