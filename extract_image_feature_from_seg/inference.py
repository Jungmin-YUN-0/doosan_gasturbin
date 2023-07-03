import os
import numpy as np
import pickle
import dill
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor

import segmentation_models_pytorch as smp
from utils.accuracy import accuracy_check
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser()
        
parser.add_argument('-data_option', help="in792sx | in792sx_interrupt | cm939w")
parser.add_argument('-feature_num', help="1 ~ 6")

opt = parser.parse_args()

'''
[1] "option" _원하는 데이터 설정
[2] 경로설정
  - "data_root_path" _데이터셋 위치
  - "output_dir" _결과 저장 위치
[3] "feature_num" _이미지 피쳐값 설정
'''

def main():
    option = opt.data_option
    feature_num = opt.feature_num

    # option = 'in792sx'
    # option = 'in792sx_interrupt'
    # option = 'cm939'
    
    ## 데이터셋 위치 설정
    if option == 'in792sx' :   
        data_root_path = '/HDD/dataset/doosan/tmp/'    
        image_path = '/HDD/dataset/doosan/tmp/images/'    # for in792sx
        model_path = '/home/tnwls/code/gasturbin/tnwls/gamma_model_best.pt'    # for in792sx
    elif option == 'in792sx_interrupt' :
        #data_root_path = '/HDD/dataset/doosan/IN792sx_interrupt/'
        data_root_path = '/HDD/jungmin/doosan/interrupt_add'
        #image_path = '/HDD/dataset/doosan/IN792sx_interrupt/images/'    # for in792sx_interrupt
        image_path = '/HDD/jungmin/doosan/interrupt_add/Interrupt_Image'
        model_path = '/HDD/tnwls/doosan/history/230227/IN792sx_interrupt/resnet18_4_32/saved_models/model_best.pt'    # for in792sx_interrupt 
    elif option == 'cm939w':
        #data_root_path = '/HDD/dataset/doosan/CM939W/'
        data_root_path = '/HDD/jungmin/doosan/cm939_add'
        #image_path = '/HDD/dataset/doosan/CM939W/images/'    # for cm929w    
        image_path = '/HDD/jungmin/doosan/cm939_add/0324' # 3/24 updated version
        #model_path = '/HDD/tnwls/doosan/history/230302/CM939W/resnet18_4_32_3/saved_models/model_best.pt'    # for cm929w
        model_path = '/HDD/tnwls/doosan/history/230324/CM939W/model_best.pt' #3/24 updated version

    if option == 'in792sx' :   
        with open(os.path.join('/HDD/jungmin/doosan', 'images.txt')) as f:
            lines = f.readlines()
        data_list = [line.rstrip('\n') for line in lines]
    else:    
        with open(os.path.join(data_root_path, 'images.txt')) as f:
            lines = f.readlines()
        data_list = [line.rstrip('\n') for line in lines]
            
    # data_root_path = '/HDD/dataset/doosan/IN792sx_interrupt/'                
    # mask_path = '/HDD/dataset/doosan_tnwls/masked_images'
    # model_path = '/HDD/tnwls/doosan/history/221009/gamma/resnet34_4_32_aug_rand/saved_models'
         
    # with open(os.path.join(data_root_path, 'images.txt'), 'rb') as f:
    #     data_list = pickle.load(f)
    #     if option == 'in792sx_interrupt':
    #         data_list = data_list[:-1] # interrupt의 경우 필요    
            
    # save_dir = os.path.join('/HDD/jungmin/doosan/history/', 'inx_inference_images/')  #[2]
    # # save_dir = os.path.join(image_path, 'inference_images/')
    # print(save_dir)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    if option == 'in792sx' :   
        model = smp.DeepLabV3('resnet34', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)
    elif option == 'in792sx_interrupt' :
        model = smp.DeepLabV3('resnet18', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)
    elif option == 'cm939w':
        model = smp.DeepLabV3('resnet18', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)
    
    #model = smp.DeepLabV3('resnet18', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)
    #model = smp.Unet('vgg16_bn', in_channels=1, classes=1)
    #model = smp.DeepLabV3Plus('timm-regnety_320',  encoder_depth=5, encoder_output_stride=16, decoder_channels=512, in_channels=1) #relu encoder_depth=4,
    #model = smp.DeepLabV3('resnet34', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)

    model.load_state_dict(torch.load(os.path.join(model_path)))

    model.cuda()

    ## Train
    total_acc = 0
    toTensor = ToTensor() 
    minmax_scaler = MinMaxScaler()
    model.eval()

    image_name = []
    feature_output=[]

    transform = transforms.Compose([transforms.Resize((512,512)),
        transforms.ToTensor(),
     ])
    
    for data in data_list:
        data = data.split(' ')[0]
        img = Image.open(os.path.join(image_path, data))
        # msk = Image.open(os.path.join(mask_path, data))
        
        # img = toTensor(img).cuda().unsqueeze(0)
        # # msk = toTensor(msk).cuda().unsqueeze(0)
        img = transform(img).cuda().unsqueeze(0)
        #msk = transform(msk).cuda().unsqueeze(0)
        
        image_name.append(data)

        with torch.no_grad():
            output, features, tmp = model(img)
            
            decoder_output = tmp
            
            ## feature 및 decoder_output 추출 후 저장
            if feature_num == "1":
                feature_output.append(features[0])
                feature_name = "f1"
            elif feature_num == "2":
                feature_output.append(features[1])
                feature_name = "f2"
            elif feature_num == "3":
                feature_output.append(features[2])
                feature_name = "f3"
            elif feature_num == "4":
                feature_output.append(features[3])
                feature_name = "f4"
            elif feature_num == "5":
                feature_output.append(features[4])
                feature_name = "f5"
            elif feature_num == "6":
                feature_output.append(decoder_output)
                feature_name = "decoder_output"
    
    '''
    ## 메모리 이슈 발생 시, 따로 저장
    data = {
        'image_name' : image_name,
        'features1' : f1,
        'features2' : f2,
        'features3' : f3,
        'feature_output' : feature_output
        'features5' : f5,
        'decoder_output' : dec
    }
    '''
    
    data={
         'image_name' : image_name,
         'feature_output':feature_output,
           }
    
    ## 결과 저장 위치 설정
    output_dir = f'/home/jungmin/workspace/doosan/image_features_{str(option)}_{str(feature_name)}.pkl'
    print('[Info] Dumping the image features to pickle file')
    dill.dump(data, open(output_dir, 'wb'))
    print('[Info] Done..')
    
        #print(features)

        # print(output.shape)
        # result = minmax_scaler.fit_transform(output[0][0].cpu())
        # result[result < 0.5] = 0
        # result[result >= 0.5] = 255

        # # mask_ori = minmax_scaler.fit_transform(msk[0][0].cpu())
        # # mask_ori[mask_ori < 0.5] = 0
        # # mask_ori[mask_ori >= 0.5] = 255

        # palette = [0,0,0, 255,255,255]
        # out = Image.fromarray(result.astype(np.uint8), mode='P')
        # out.putpalette(palette)

        # export_name = str(data)
        # out.save(save_dir + export_name)
        # acc = accuracy_check(mask_ori, out)
        # total_acc += acc

        # fig = plt.figure()
        # rows = 1
        # cols = 2

        # ax1 = fig.add_subplot(rows, cols, 1)
        # ax1.imshow(img[0][0].cpu().numpy(), 'gray')
        # ax1.set_title('original image')
        # ax1.axis("off")

        # ax2 = fig.add_subplot(rows, cols, 2)
        # ax2.imshow(result.astype(np.uint8), 'gray')
        # ax2.set_title('prediction')
        # ax2.axis("off")

        # # ax3 = fig.add_subplot(rows, cols, 3)
        # # ax3.imshow(mask_ori, 'gray')
        # # ax3.set_title('ground truth')
        # # ax3.axis("off")

        # export_name_fig = export_name[:-4] +'_fig.png'
        # plt.savefig(save_dir + export_name_fig)
    
    # print('total_acc : ', total_acc/len(data_list))
        

        

    # total_acc = inference(image_path, mask_path, model, save_dir)
    
    # print('TOTAL ACC!!', total_acc)
if __name__ == "__main__":
    main()
