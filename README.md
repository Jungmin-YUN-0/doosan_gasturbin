# doosan_gasturbin

<b> [1] extract_iamge_feature_from_seg/inference.py </b>

: 이미지 피쳐값 추출

(1) args
  - "option" _원하는 데이터 지정
  - "feature_num" _이미지 피쳐값 설정
    
(2) dir
  - "data_root_path" _데이터셋 경로
  - "output_dir" _결과 저장 경로

<b> [2] merge_image_feature.py </b>

: 시험정보 & 야금학적 특징 통합

(1) args
  - "data_dir" _시험정보 데이터 경로
  - "in792sx_dir" _in792sx 야금학적 특징 데이터 경로
  - "in792sx_interrupt_dir"_in792sx interrupt 야금학적 특징 데이터 경로
  - "cm939w_dir" _cm939w 야금학적 특징 데이터 경로
  - "save_dir"_결과 저장 경로
    
<b> [3] data_preprocess_step2.ipynb </b>

: 시험정보 & 야금학적 특징 통합
 
- input: gasturbin_data.csv
- output: data_all_feature.csv


