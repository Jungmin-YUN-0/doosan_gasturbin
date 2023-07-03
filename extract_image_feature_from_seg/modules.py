import numpy as np
from PIL import Image
import csv
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torch.nn as nn

from dataset import *
from utils.accuracy import accuracy_check, accuracy_check_for_batch
from sklearn.metrics import accuracy_score


def train_model(i, model, data_train, fn_loss, optimizer):
    """Train the model and report validation error with training error
    Args:
        model: the model to be trained
        criterion: loss function
        data_train (DataLoader): training dataset
    """
    total_acc = 0
    total_loss = 0
    ba = 0
    model.train()
    for batch, (name, images, masks) in enumerate(data_train):
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())
       
        b, h, w = masks.shape
        tmp_masks = (masks.clone().detach().view(b,1,h,w)).type(torch.float)
        
        outputs, _, _ = model(images)
        
        loss = fn_loss(outputs, tmp_masks)
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()
        acc = accuracy_check_for_batch(tmp_masks.cpu(), outputs.cpu(), images.size()[0])
        total_acc = total_acc + acc
        total_loss = total_loss + loss.cpu().item()

        if batch%100 == 0:
            print('Epoch', str(i+1), 'Train loss:', total_loss, "Train acc", total_acc)

    return total_loss, total_acc

def train_model_cls(i, model, data_train, fn_loss, optimizer):
    """Train the model and report validation error with training error
    Args:
        model: the model to be trained
        criterion: loss function
        data_train (DataLoader): training dataset
    """
    total_acc = 0
    total_loss = 0
    ba = 0
    model.train()
    for batch, (name, images, label) in enumerate(data_train):
        images = Variable(images.cuda())
        label = Variable(label.cuda())
       
        
        outputs = model(images)
        
        loss = fn_loss(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()
        # print(label.cpu().numpy())
        # print(outputs.cpu().detach().numpy()>0.5)
        acc = accuracy_score(label.cpu().numpy(), outputs.cpu().detach().numpy()>0.5)
        # print(acc)
        total_acc = total_acc + acc
        
        total_loss = total_loss + loss.cpu().item()
        ba+=batch
        

        if batch%100 == 0:
            print('Epoch', str(i+1), 'Train loss:', total_loss/(ba+1), "Train acc", total_acc/(ba+1))

    return total_loss, total_acc

def get_loss_train(model, data_train, fn_loss):
    """
        Calculate loss over train set
    """
    model.eval()
    total_acc = 0
    total_loss = 0
    for batch, (images, masks) in enumerate(data_train):
        with torch.no_grad():
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())
            b, h,w = masks.shape
            outputs = model(images)

            # loss = fn_loss(outputs, masks)
            loss = fn_loss(outputs, torch.tensor(masks, dtype=torch.float).view(b, 1, h, w))

            # preds = torch.argmax(outputs, dim=1).float()
            acc = accuracy_check_for_batch(masks.cpu(), outputs.cpu(), images.size()[0])
            total_acc = total_acc + acc
            total_loss = total_loss + loss.cpu().item()
    return total_acc/(batch+1), total_loss/(batch + 1)


def validate_model(model, data_val, criterion, epoch, make_prediction=True, save_dir='prediction'):
    """
        Validation run
    """
    # calculating validation loss
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    minmax_scaler = MinMaxScaler()
    for batch, (images_v, masks_v, original_msk, name) in enumerate(data_val):
        stacked_img = torch.Tensor([]).cuda()
        # for index in range(images_v.size()[1]):
        with torch.no_grad():
            image_v = Variable(images_v.unsqueeze(0).cuda())
            label = Variable(label.squeeze(1).cuda())
            
        
            b, h, w = mask_v.shape
            # print(image_v.shape)
            # print(mask_v.shape)
            tmp_mask = (mask_v.clone().detach().view(b,1,h,w)).type(torch.float)
            output_v = model(image_v)
            # print(output_v.shape)
    
            tmp = criterion(output_v, tmp_mask)
            total_val_loss = total_val_loss + tmp.cpu().item()
            # print('out', output_v.shape)
            # output_v = torch.argmax(output_v, dim=1).float()
            stacked_img = torch.cat((stacked_img, output_v.view(b,h,w)))

        if make_prediction:
            im_name = name  # TODO: Change this to real image name so we know
            pred_msk = save_prediction_image(stacked_img, im_name, epoch, save_dir)

            gt_msk = minmax_scaler.fit_transform(tmp_mask.cpu()[0][0])
            gt_msk[gt_msk < 0.5] = 0
            gt_msk[gt_msk >= 0.5] = 255


            acc_val = accuracy_check(gt_msk, pred_msk)
            total_val_acc = total_val_acc + acc_val

    return total_val_acc/(batch + 1), total_val_loss/((batch + 1)*4)



def validate_model_cls(model, data_val, criterion, epoch, make_prediction=True, save_dir='prediction'):
    """
        Validation run
    """
    # calculating validation loss
    model.eval()
    total_loss = 0
    total_acc = 0
    # minmax_scaler = MinMaxScaler()
    for batch, (images, label) in enumerate(data_val):
        # stacked_img = torch.Tensor([]).cuda()
        # for index in range(images_v.size()[1]):
        with torch.no_grad():
            # print(images.shape)
            images = Variable(images.unsqueeze(0).cuda())
            label = Variable(label.cuda())
        
            
            outputs = model(images)
            
            loss = criterion(outputs, label)
            
            acc = accuracy_score(label.cpu().numpy(), outputs.cpu().detach().numpy()>0.5)
            total_acc += acc
            
            total_loss = total_loss + loss.cpu().item()

    return total_acc/len(data_val), total_loss/len(data_val)

def save_prediction_image(stacked_img, im_name, epoch, save_dir="result_images", save_im=True):
    """save images to save_path
    Args:
        stacked_img (numpy): stacked cropped images
        save_folder_name (str): saving folder name
    """

    save_folder_name = os.path.join(save_dir, 'result_images')
    # div_arr = division_array(388, 2, 2, 512, 512)
    # img_cont = image_concatenate(stacked_img.cpu().data.numpy(), 2, 2, 512, 512)
    minmax_scaler = MinMaxScaler()
    result = minmax_scaler.fit_transform(stacked_img[0].cpu())
    result[result < 0.5] = 0
    result[result >= 0.5] = 255

    # img_cont = polarize((img_cont)/div_arr)*255
    # img_cont_np = img_cont.astype('uint8')
    # img_cont = Image.fromarray(img_cont_np)
 

    palette = [0,0,0, 255,255,255]
    out = Image.fromarray(result.astype(np.uint8), mode='P')
    out.putpalette(palette)
    # organize images in every epoch
    desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(im_name[0])
    out.save(desired_path + export_name)
    return out

def save_prediction_image2(img, mask, pred, im_name, save_folder_name="result_images"):
    """save images to save_path
    Args:
        stacked_img (numpy): stacked cropped images
        save_folder_name (str): saving folder name
    """
    img
    minmax_scaler = MinMaxScaler()
    result = minmax_scaler.fit_transform(img_cont)
    result[result < 0.5] = 0
    result[result >= 0.5] = 255

    # img_cont = polarize((img_cont)/div_arr)*255
    # img_cont_np = img_cont.astype('uint8')
    # img_cont = Image.fromarray(img_cont_np)
 

    palette = [0,0,0, 255,255,255]
    out = Image.fromarray(result.astype(np.uint8), mode='P')
    out.putpalette(palette)
    # organize images in every epoch
    desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(im_name[0])
    out.save(desired_path + export_name)
    return out


def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img


"""
def test_SEM(model, data_test,  folder_to_save):
    '''Test the model with test dataset
    Args:
        model: model to be tested
        data_test (DataLoader): test dataset
        folder_to_save (str): path that the predictions would be saved
    '''
    for i, (images) in enumerate(data_test):

        print(images)
        stacked_img = torch.Tensor([])
        for j in range(images.size()[1]):
            image = Variable(images[:, j, :, :].unsqueeze(0).cuda())
            output = model(image.cuda())
            print(output)
            print("size", output.size())
            output = torch.argmax(output, dim=1).float()
            print("size", output.size())
            stacked_img = torch.cat((stacked_img, output))
        div_arr = division_array(388, 2, 2, 512, 512)
        print(stacked_img.size())
        img_cont = image_concatenate(stacked_img.data.numpy(), 2, 2, 512, 512)
        final_img = (img_cont*255/div_arr)
        print(final_img)
        final_img = final_img.astype("uint8")
        break
    return final_img
"""


if __name__ == '__main__':
    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    SEM_train_load = torch.utils.data.DataLoader(dataset=SEM_train,
                                                 num_workers=3, batch_size=10, shuffle=True)
    get_loss_train()
