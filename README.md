# doosan_gasturbin

<b> [1] extract_iamge_feature_from_seg/inference.py </b>

: 이미지 피쳐값 추출

(1) args
  - "option" _원하는 데이터 지정
  - "feature_num" _이미지 피쳐값 설정
    
(2) dir
  - "data_root_path" _데이터셋 경로
  - "output_dir" _결과 저장 경로

<b> [2] data_preprocessing.py </b>

: 시험정보 & 야금학적 특징 통합

- input: gasturbin_data.csv (원본 시험정보 데이터 - in792sx, in792sx interrupt, cm939w 통합본)
- input: in792sx_features.csv, in792sx_interrupt_features.csv, cm939w_features.csv (야금학적 물성정보 데이터)
- output: data_all_feature.csv
(1) args
  - "data_dir" _시험정보 데이터 경로 (gasturbin_data.csv)
  - "in792sx_dir" _in792sx 야금학적 특징 데이터 경로
  - "in792sx_interrupt_dir"_in792sx interrupt 야금학적 특징 데이터 경로
  - "cm939w_dir" _cm939w 야금학적 특징 데이터 경로 
  - "save_dir"_결과 저장 경로 (data_all_feature.csv)
    
<b> [3] merge_image_features.py </b>

: 이미지 피쳐맵 간 통합

- input: image_features_{option}_{feature_num}.pkl
- output: image_feature_processed_{option}.csv


- input: data_all_feature.csv (시험정보, 야금학적 물성정보)
- input: image_features_{option}_{feature_num}.pkl (이미지 피쳐맵)
- output: data_all_features_add_image.csv (최종 통합본)




