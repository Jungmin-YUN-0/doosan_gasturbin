import dill
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument('-option', help="in792sx | in792sx_interrupt | cm939w")
opt = parser.parse_args()


class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=8)
        self.emb = nn.Linear(64,1)
    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = x.view(1,32,-1)
        x = x.squeeze()
        x = self.emb(x)
        x = x.permute(1,0)
        return x
    
class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=16, stride=8)
        self.emb = nn.Linear(217,1)
    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = x.view(1,32,-1)
        x = x.squeeze()
        x = self.emb(x)
        x = x.permute(1,0)
        return x
    
class CNN_3(nn.Module):
    def __init__(self):
        super(CNN_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=16, stride=8)
        self.emb = nn.Linear(105,1)
    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = x.view(1,32,-1)
        x = x.squeeze()
        x = self.emb(x)
        x = x.permute(1,0)
        return x
    
class CNN_4(nn.Module):
    def __init__(self):
        super(CNN_4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=16, stride=8)
        self.emb = nn.Linear(105,1)
    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = x.view(1,32,-1)
        x = x.squeeze()
        x = self.emb(x)
        x = x.permute(1,0)
        return x
    
class CNN_5(nn.Module):
    def __init__(self):
        super(CNN_5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=16, stride=8)
        #self.conv2 = nn.Conv2d(in_channels=80, out_channels=32, kernel_size=8, stride=4)
        self.emb = nn.Linear(217,1)
    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = x.view(1,32,-1)
        x = x.squeeze()
        x = self.emb(x)
        x = x.permute(1,0)
        return x
    
class CNN_6(nn.Module):
    def __init__(self):
        super(CNN_6, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=16, stride=8)
        self.emb = nn.Linear(21,1)
    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = x.view(1,32,-1)
        x = x.squeeze()
        x = self.emb(x)
        x = x.permute(1,0)
        return x
    
#data = dill.load(open(r'/home/jungmin/workspace/doosan/image_features_in792sx.pkl', 'rb'))
#data = dill.load(open(r'/home/jungmin/workspace/doosan/image_features_in792sx_interrupt.pkl', 'rb'))

def feature_process(option, feature_num):
  data = dill.load(open(f'/home/jungmin/workspace/doosan/image_features_{option}_{feature_num}.pkl', 'rb'))
  #print(len(data['feature_output']), len(data['image_name']))

  if feature_num == "f1" :
    cnn = CNN_1()
  elif feature_num == 'f2':
    cnn = CNN_2()
  elif feature_num == 'f3':
    cnn = CNN_3()
  elif feature_num == 'f4':
    cnn = CNN_4()
  elif feature_num == 'f5':
    cnn = CNN_5()
  elif feature_num == 'decoder_output':
    cnn = CNN_6()
  cnn.cuda()
  #output = cnn(data['feature_output'][1])  # Input Size: (10, 1, 20, 20)
  processed = []
  for i in tqdm(range(0,len(data['feature_output']))):
    processed.append(cnn(data['feature_output'][i]).tolist())
  image_name = data['image_name']
  return processed, image_name    
    

#-------------------------------------------------
f1 = []
f2 = []
f3 = []
f4 = []
f5 = []
f6 = []
#-------------------------------------------------
f1, image_name = feature_process(option, "f1")
f2, _ = feature_process(option, "f2")
f3, _ = feature_process(option, "f3")
f4, _ = feature_process(option, "f4")
f5, _ = feature_process(option, "f5")
f6, _ = feature_process(option, "decoder_output")

import pandas as pd
df2 = pd.DataFrame([image_name, f1, f2, f3, f4, f5, f6]).T
df2.columns=['image_name','image_feature_1','image_feature_2','image_feature_3','image_feature_4','image_feature_5','image_feature_6']

df2.to_csv(f'/home/jungmin/workspace/doosan/image_feature_processed_{option}.csv',index=False,sep=',',encoding='utf-8')

df2.tail()
    
    
    