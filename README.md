# SPMVF
This code accompanies the paper: Spatial Puzzle-based with Multi-level Visual Features for Self-Supervised Video Anomaly Detection



#### Requirements :
    conda env create -f environment.yml
    conda activate env
    

#### Datasets :
* UCSD Ped2: http://www.svcl.ucsd.edu/projects/anomaly
* CUHK Avenue: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
* Shanghaitech: https://svip-lab.github.io/dataset/campus_dataset.html
* Video formats can be processed and named from tools/video2image.py and tools/video2image.py.


#### Preparation
```
python crop.py --dataset ped2 --phase train/test --filter_ratio 0.6 --sample_num 7
```

#### Training :
    python train.py --dataset ped2 --batch_size 256 --sample_num 7 --epochs 120

#### Inference :
    python val.py --dataset ped2 --sample_num 7 --checkpoint xxx

#### Benchmark :
* Use tools/eval.py

