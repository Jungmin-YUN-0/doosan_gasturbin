import argparse 
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable

from models.advanced_model import CleanU_Net
from dataset import CustomDatasetTrain, CustomDatasetVal
from modules import train_model, validate_model
from utils.save_history import export_history, save_models
from utils.util import train_val_split_

import segmentation_models_pytorch as smp

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def main(args):
    
    train_val_split_(args.data_root_path)

    trainset = CustomDatasetTrain(args.image_path, args.mask_path)
    valset = CustomDatasetVal(args.image_path, args.mask_path) 

    save_path = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    SEM_train_load = torch.utils.data.DataLoader(dataset=trainset,
                                                 num_workers=6, 
                                                 batch_size=16, 
                                                 shuffle=True)
    SEM_val_load = torch.utils.data.DataLoader(dataset=valset,
                                               num_workers=6, 
                                               batch_size=1, 
                                               shuffle=False)


    # model = CleanU_Net(in_channels=1, out_channels=1)
    # model = smp.DeepLabV3Plus('timm-regnety_320',  encoder_depth=5, encoder_output_stride=16, decoder_channels=512, in_channels=1) #relu encoder_depth=4,
    # model = smp.DeepLabV3('resnet18', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)
    model = smp.DeepLabV3('resnet34', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)
    # model = smp.Unet('vgg16_bn', in_channels=1, classes=1)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    epoch_start = 0
    epoch_end = 500

    # Saving History to csv
    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
 

    fn_loss = nn.BCEWithLogitsLoss()

    # Train
    best_val_acc = 0
    print("Initializing Training!")
    for i in range(epoch_start, epoch_end):
        # train the model
        train_loss, train_acc = train_model(i, model, SEM_train_load, fn_loss, optimizer)
        
        # # # # # Validation every 5 epoch
        if (i+1) % 5 == 0:
            val_acc, val_loss = validate_model(
                model, SEM_val_load, fn_loss, i+1, True, save_path)
            print('Val loss:', val_loss, "val acc:", val_acc)
            values = [i+1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_path)

            
            if best_val_acc <= val_acc :  # save model every 10 epoch
                best_val_acc = val_acc
                save_models(model, save_path, i+1, True)
        
        if (i+1) % 10 == 0:  # save model every 10 epoch
            save_models(model, save_path, i+1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    parser.add_argument('--data_root_path', type=str, default='/HDD/dataset/doosan/tmp/')
    parser.add_argument('--image_path', type=str, default='/HDD/dataset/doosan/tmp/images')
    parser.add_argument('--mask_path', type=str, default='/HDD/dataset/doosan/tmp/inference_images')
    parser.add_argument('--save_dir', type=str, default='/HDD/jungmin/doosan/history/gamma/')
    parser.add_argument('--model_name', type=str, default='resnet34_4_32_semi')

    args = parser.parse_args()



    main(args)
