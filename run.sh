# Instruction of how to run your own dataset
# python data_preprocess.py /storage/company_old/*/*.wav
# python grid_search.py

# test run
export CUDA_VISIBLE_DEVICES=0
python data_preprocess.py datasets/company_old "preprocessed_company_data" 8000
python train.py "preprocessed_company_data"


