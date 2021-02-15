
#!/bin/bash

mkdir out
# Put boston_housing.arff file in data/100000 folder. You can rename it as 100000.arff for consistency
python cafem.py --multiprocessing 6 --out_dir   out/ml --num_episodes 5
python cafem.py --out_dir   out/ml --cuda 0 --num_epochs 1 --meta_batch_size 1 --num_episodes 1

# Passing 100000 as command line argument to predict on boston_housing data
python single_afem.py --load_weight   out/ml/model/model_5.ckpt --dataset 100000 --out_dir   out/o2_5_1049 --num_epochs 50 --buffer_size 1000 --num_episodes 1
python single_afem.py --load_weight   out/ml/cafem/model_1.ckpt --dataset 100000 --out_dir   out/o2_5_1049 --num_epochs 1 --buffer_size 1000 --num_episodes 1
