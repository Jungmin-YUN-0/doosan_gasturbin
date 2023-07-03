import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import pi


def cal_rect(contours_xy):
    cnt = contours_xy
    
    rect = cv2.minAreaRect(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    return box, x,y,w,h

def cal_distrib(gamma_prime):
    distrib = None

    if gamma_prime <= 2538:
        distrib = 0
    elif 2538 < gamma_prime and gamma_prime <=7614:
        distrib = 1
    elif 7614 < gamma_prime and gamma_prime <=25380:
        distrib = 2
    elif 25380 < gamma_prime and gamma_prime <=76142:
        distrib = 3
    elif 76142 < gamma_prime:
        distrib = 4            
    return distrib

def cal_width(img, gamma_prime, b):
    h, w, _ = img.shape
    return b*gamma_prime/(h*w)

def cal_circle(img, gamma_prime):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
    contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = len(contour[0])
    
#     img3 = cv2.drawContours(img1, [cnt], 0, (0,255,0), 1)    
    
    return (4*pi*gamma_prime)/(perimeter*perimeter)

if __name__ == "__main__":
    image_name = 'tnwls2.png' 
    name = image_name[:-4]
    save_path = './' + name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
            
    img = cv2.imread(image_name)
    img_cp = img.copy()
    imgray = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
    ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
    contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img_h, img_w, _  = img.shape
    img_flatten = imgray.flatten()
    gamma = len(np.where(img_flatten==255)[0])
    gamma_prime = len(np.where(img_flatten==0)[0])

    trims_coord = []
    gamma_size_distrib = []
    gamma_ratios = []
    gamma_bboxes=[]
    gamma_W= []
    gamma_circle = []
    gamma_size = []
    # img_h, img_w, _ = img.shape
    for i in range(len(contour)):
        
        contours_xy = np.array(contour[i])
        box, x, y, w, h = cal_rect(contours_xy)

        img_rect = cv2.drawContours(img_cp,[box],0,(0,0,255),2)
        img_tmp = np.ones((img_h, img_w, 3), dtype=np.uint8)*255
        
        trim_fill_img = cv2.fillPoly(img_tmp,[contours_xy], color = (0,0,0))
        img_trim = trim_fill_img[y:y+h, x:x+w]
        
        trim_gray = cv2.cvtColor(img_trim, cv2.COLOR_BGR2GRAY)
        trim_flatten = trim_gray.flatten()
        trim_gamma_prime = len(np.where(trim_flatten==0)[0])
        gamma_size.append(trim_gamma_prime)
        
        
        distrib = cal_distrib(trim_gamma_prime)
        gamma_size_distrib.append(distrib)
    
        
        trim_h, trim_w, _ = img_trim.shape
        if trim_h > trim_w:
            a = trim_h
            b = trim_w
        else:
            a = trim_w
            b = trim_h
            
        trim_ratio = a/b
        
        gamma_bboxes.append(b*(trim_gamma_prime/(trim_h*trim_w)))
        gamma_ratios.append(trim_ratio*trim_gamma_prime)
        
        trimW = cal_width(img_trim, trim_gamma_prime, b)     
        gamma_W.append(trimW*trim_gamma_prime)
        
        trimC = cal_circle(img_trim, trim_gamma_prime)
        gamma_circle.append(trimC*trim_gamma_prime)

        cv2.imwrite(os.path.join(save_path, 'gamma_%s.png'%format(i)), img_trim)

    cv2.imwrite(os.path.join(save_path, 'gamma_all.png'), img_rect)

    aspect_ratio = 0
    avg_width = 0
    avg_cir = 0
    area_0, area_1, area_2, area_3, area_4 = 0, 0, 0, 0, 0
    for i in range(len(gamma_size_distrib)):
        aspect_ratio += gamma_ratios[i]
        avg_width += gamma_W[i]
        avg_cir += gamma_circle[i]
        
        if gamma_size_distrib[i] == 0:
            area_0 += 1
        elif gamma_size_distrib[i] == 1:
            area_1 += 1
        elif gamma_size_distrib[i] == 2:
            area_2 += 1
        elif gamma_size_distrib[i] == 3:
            area_3 += 1
        else: area_4+= 1

            
    print('Gamma Phase  : ', (gamma/len(img_flatten))*100, '%')
    print('Gamma Prime Phase: ',(gamma_prime/len(img_flatten))*100, '%')

    print('=========================================')
    print('Gamma Prime Size Distribution')
    print('    area  <=1  :', round(area_0/len(gamma_size_distrib)*100,3),"%")
    print('1  < area <=3  :', round(area_1/len(gamma_size_distrib)*100,3),"%")
    print('3  < area <=10 :', round(area_2/len(gamma_size_distrib)*100,3),"%")
    print('10 < area <=30 :', round(area_3/len(gamma_size_distrib)*100,3),"%")
    print('30 < area      :', round(area_4/len(gamma_size_distrib)*100,3),"%")
    print('=========================================')
    print('Gamma Prime Aspect Ratio : ', aspect_ratio/gamma_prime)
    print('Gamma Prime Average Width : ', avg_width/gamma_prime)
    print('Gamma Prime Circulairty : ', avg_cir/gamma_prime)