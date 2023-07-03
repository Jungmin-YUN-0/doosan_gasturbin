import pandas as pd
import argparse

parser = argparse.ArgumentParser()
        
parser.add_argument('-data_dir', required=True, help="data path")
parser.add_argument('-in792sx_dir', required=True, help="in792sx image feature data path")
parser.add_argument('-in792sx_interrupt_dir', required=True, help="in792sx_interrupt image feature data path")
parser.add_argument('-cm939w_dir', required=True, help="cm939w image feature data path")
parser.add_argument('-save_dir', required=True, help="data save path")

opt = parser.parse_args()

#data_path = r'/home/jungmin/workspace/doosan/doosan_result/gasturbin_data.csv'
df = pd.read_csv(data_path, encoding='UTF-8', sep=',') #IN792sx, interrupt, cm939 통합 버전
df = df[['file','test_id','temp_oc','stress_mpa','LMP','mean','lower','upper']]
df.columns = ['file','id','temp_oc','stress_mpa','LMP','mean','lower','upper']


#################
#### in792sx ####
#################
#in792sx_dir = r'/HDD/jungmin/doosan/in793sx_features.csv'
in792xs = pd.read_csv(in792sx_dir, encoding='UTF-8', sep=',') #IN792sx, interrupt 합친 데이터
in792xs['id'] = in792xs['Name'].str[7:-14]

# string preprocess
in792xs['id'] = in792xs['id'].replace('409','A0409',regex=True)
in792xs['id'] = in792xs['id'].replace('410','A0410',regex=True)
in792xs['id'] = in792xs['id'].replace('411','A0411',regex=True)
in792xs['id'] = in792xs['id'].replace('412','A0412',regex=True)
in792xs['id'] = in792xs['id'].replace('AC1','A-C1',regex=True)
in792xs['id'] = in792xs['id'].replace('AC2','A-C2',regex=True)
in792xs['id'] = in792xs['id'].replace('AC3','A-C3',regex=True)
in792xs['id'] = in792xs['id'].replace('AC4','A-C4',regex=True)
in792xs['id'] = in792xs['id'].replace('AC5','A-C5',regex=True)
in792xs['id'] = in792xs['id'].replace('AC6','A-C6',regex=True)
in792xs['id'] = in792xs['id'].replace('AC7','A-C7',regex=True)
in792xs['id'] = in792xs['id'].replace('AC8','A-C8',regex=True)
in792xs['id'] = in792xs['id'].replace('AC9','A-C9',regex=True)
in792xs['id'] = in792xs['id'].replace('AC10','A-C10',regex=True)
in792xs['id'] = in792xs['id'].replace('AC11','A-C11',regex=True)
in792xs['id'] = in792xs['id'].replace('AC12','A-C12',regex=True)

in792xs = in792xs[['id','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]


###########################
#### in792sx_interrupt ####
###########################
#in792sx_interrupt_dir = r'/HDD/jungmin/doosan/interrupt_add/features_in792sx_interrupt.csv'
interrupt = pd.read_csv(in792sx_interrupt_dir, encoding='UTF-8', sep=',') #IN792sx, interrupt 합친 데이터

# string preprocess
interrupt['id'] = interrupt['Name'].str[:4]
interrupt['id'] = interrupt['id'].replace('9_1_','9_1', regex=True)
interrupt['id'] = interrupt['id'].replace('9_2_','9_2', regex=True)
interrupt['id'] = interrupt['id'].replace('9_3_','9_3', regex=True)
interrupt['id'] = interrupt['id'].replace('9_4_','9_4', regex=True)
interrupt['id'] = interrupt['id'].replace('9_5_','9_5', regex=True)
interrupt['id'] = interrupt['id'].replace('7_1_','7_1', regex=True)
interrupt['id'] = interrupt['id'].replace('7_2_','7_2', regex=True)
interrupt['id'] = interrupt['id'].replace('7_3_','7_3', regex=True)
interrupt['id'] = interrupt['id'].replace('7_4_','7_4', regex=True)
interrupt['id'] = interrupt['id'].replace('7_5_','7_5', regex=True)

interrupt = interrupt[['id','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]


################
#### cm939w ####
################
cm939w_dir = r'/HDD/dataset/doosan/CM939W/features.csv'
cm939w = pd.read_csv(cm939w, encoding='UTF-8', sep=',', header=None) #IN792sx, interrupt 합친 데이터
#cm939w.columns=['idx','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']

cm939w['id'] = cm939w[1].replace('.png','',regex=True)
cm939w['id'] = cm939w['id'].replace('900_','',regex=True)
cm939w['id'] = cm939w['id'].replace('950_','',regex=True)
cm939w['id'] = cm939w['id'].replace('1000_','',regex=True)
cm939w['id'] = cm939w['id'].replace('_1','',regex=True)
cm939w['id'] = cm939w['id'].replace('_2','',regex=True)
cm939w['id'] = cm939w['id'].replace('_3','',regex=True)
cm939w['id'] = cm939w['id'].replace('_4','',regex=True)
cm939w['id'] = cm939w['id'].replace('_5','',regex=True)
cm939w['id'] = cm939w['id'].replace('_6','',regex=True)
cm939w['id'] = cm939w['id'].replace('_7','',regex=True)
cm939w['id'] = cm939w['id'].replace('_8','',regex=True)
cm939w['id'] = cm939w['id'].replace('_9','',regex=True)
cm939w['id'] = cm939w['id'].replace('_10','',regex=True)
cm939w['id'] = cm939w['id'].replace('g','G',regex=True)
cm939w['id'] = cm939w['id'].replace('c','C',regex=True)
cm939w['id'] = cm939w['id'].replace('_','-',regex=True)
cm939w['id'] = cm939w['id'].replace('C110','C11',regex=True)
cm939w['id'] = cm939w['id'].replace('C111','C11',regex=True)
cm939w['id'] = cm939w['id'].replace('C112','C11',regex=True)
cm939w['id'] = cm939w['id'].replace('C113','C11',regex=True)
cm939w['id'] = cm939w['id'].replace('C114','C11',regex=True)
cm939w['id'] = cm939w['id'].replace('C115','C11',regex=True)
cm939w['id'] = cm939w['id'].replace('C116','C11',regex=True)
cm939w['id'] = cm939w['id'].replace('C117','C11',regex=True)
cm939w['id'] = cm939w['id'].replace('C118','C11',regex=True)
cm939w['id'] = cm939w['id'].replace('C119','C11',regex=True)
cm939w['id'] = cm939w['id'].replace('C120','C12',regex=True)
cm939w['id'] = cm939w['id'].replace('C121','C12',regex=True)
cm939w['id'] = cm939w['id'].replace('C122','C12',regex=True)
cm939w['id'] = cm939w['id'].replace('C123','C12',regex=True)
cm939w['id'] = cm939w['id'].replace('C124','C12',regex=True)
cm939w['id'] = cm939w['id'].replace('C125','C12',regex=True)
cm939w['id'] = cm939w['id'].replace('C126','C12',regex=True)
cm939w['id'] = cm939w['id'].replace('C127','C12',regex=True)
cm939w['id'] = cm939w['id'].replace('C128','C12',regex=True)
cm939w['id'] = cm939w['id'].replace('C129','C12',regex=True)
cm939w['id'] = cm939w['id'].replace('C130','C13',regex=True)
cm939w['id'] = cm939w['id'].replace('C131','C13',regex=True)
cm939w['id'] = cm939w['id'].replace('C132','C13',regex=True)
cm939w['id'] = cm939w['id'].replace('C133','C13',regex=True)
cm939w['id'] = cm939w['id'].replace('C134','C13',regex=True)
cm939w['id'] = cm939w['id'].replace('C135','C13',regex=True)
cm939w['id'] = cm939w['id'].replace('C136','C13',regex=True)
cm939w['id'] = cm939w['id'].replace('C137','C13',regex=True)
cm939w['id'] = cm939w['id'].replace('C138','C13',regex=True)
cm939w['id'] = cm939w['id'].replace('C139','C13',regex=True)
cm939w['id'] = cm939w['id'].replace('C20','C2',regex=True)
cm939w['id'] = cm939w['id'].replace('C21','C2',regex=True)
cm939w['id'] = cm939w['id'].replace('C22','C2',regex=True)
cm939w['id'] = cm939w['id'].replace('C23','C2',regex=True)
cm939w['id'] = cm939w['id'].replace('C24','C2',regex=True)
cm939w['id'] = cm939w['id'].replace('C25','C2',regex=True)
cm939w['id'] = cm939w['id'].replace('C26','C2',regex=True)
cm939w['id'] = cm939w['id'].replace('C27','C2',regex=True)
cm939w['id'] = cm939w['id'].replace('C28','C2',regex=True)
cm939w['id'] = cm939w['id'].replace('C29','C2',regex=True)
cm939w['id'] = cm939w['id'].replace('C40','C4',regex=True)
cm939w['id'] = cm939w['id'].replace('C41','C4',regex=True)
cm939w['id'] = cm939w['id'].replace('C42','C4',regex=True)
cm939w['id'] = cm939w['id'].replace('C43','C4',regex=True)
cm939w['id'] = cm939w['id'].replace('C44','C4',regex=True)
cm939w['id'] = cm939w['id'].replace('C45','C4',regex=True)
cm939w['id'] = cm939w['id'].replace('C46','C4',regex=True)
cm939w['id'] = cm939w['id'].replace('C47','C4',regex=True)
cm939w['id'] = cm939w['id'].replace('C48','C4',regex=True)
cm939w['id'] = cm939w['id'].replace('C49','C4',regex=True)
cm939w['id'] = cm939w['id'].replace('4110','411',regex=True)
cm939w['id'] = cm939w['id'].replace('4111','411',regex=True)
cm939w['id'] = cm939w['id'].replace('4112','411',regex=True)
cm939w['id'] = cm939w['id'].replace('4113','411',regex=True)
cm939w['id'] = cm939w['id'].replace('4114','411',regex=True)
cm939w['id'] = cm939w['id'].replace('4115','411',regex=True)
cm939w['id'] = cm939w['id'].replace('4116','411',regex=True)
cm939w['id'] = cm939w['id'].replace('4117','411',regex=True)
cm939w['id'] = cm939w['id'].replace('4118','411',regex=True)
cm939w['id'] = cm939w['id'].replace('4119','411',regex=True)
cm939w['id'] = cm939w['id'].replace('4130','413',regex=True)
cm939w['id'] = cm939w['id'].replace('4131','413',regex=True)
cm939w['id'] = cm939w['id'].replace('4132','413',regex=True)
cm939w['id'] = cm939w['id'].replace('4133','413',regex=True)
cm939w['id'] = cm939w['id'].replace('4134','413',regex=True)
cm939w['id'] = cm939w['id'].replace('4135','413',regex=True)
cm939w['id'] = cm939w['id'].replace('4136','413',regex=True)
cm939w['id'] = cm939w['id'].replace('4137','413',regex=True)
cm939w['id'] = cm939w['id'].replace('4138','413',regex=True)
cm939w['id'] = cm939w['id'].replace('4139','413',regex=True)
cm939w['id'] = cm939w['id'].replace(r"prediCtion-imaGes\\",'',regex=True)
# cm939w['Name'] = cm939w['Name'].replace(r"prediction-images\\",'',regex=True)

cm939w.columns = ['idx','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle','id']
cm939w = cm939w[['id','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]


################
#### merge #####
################
in792xs['file'] = 'in792xs'
interrupt['file'] = 'interrupt'
cm939w['file'] = 'cm939w'

in792xs = in792xs[['file','id','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]
interrupt = interrupt[['file','id','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]
cm939w = cm939w[['file','id','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]

features = pd.concat([in792xs, interrupt, cm939w], axis=0)

result = pd.merge(df,features, on='id')
result = result[['file_x','id','Name','stress_mpa','temp_oc','LMP','mean','upper','lower','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]
result.rename(columns={'file_x':'file'})

#save_dir =r"/home/jungmin/workspace/doosan/data_all_feature.csv"
result.to_csv(save_dir, sep=',', index=False)