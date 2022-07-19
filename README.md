# Region-Aware Metric Learning for Open World Semantic Segmentation via Meta-Channel Aggregation

## Introduction
This is an official pytorch implementation of *Region-Aware Metric Learning for Open World Semantic Segmentation via Meta-Channel Aggregation*, IJCAI 2022. This work proposes a method called region-aware metric learning (RAML) to first separate the regions of the images and generate region-aware features for further metric learning for open world semantic segmentation. The link to the paper is [here](https://arxiv.org/abs/2205.08083).

## Quick starts

### Dataset
We follow [DMLNet](https://github.com/Jun-CEN/Open-World-Semantic-Segmentation) to prepare datasets.

Note: For different settings, you need to manually modify lines 71 through 82 in datasets/cityscapes.py.

### Pretrained model
The pretrained models can be downloaded from [Google Drive](https://drive.google.com/file/d/1GYKxToN3YzKSmx9RsDCW8A0QWFU9liZ8/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1dza_9Fr75wEKX_mmncvofA) (code: 63z1). Put four folders into RAML/incremental/.

### Training
First, go to "incremental":
```
cd incremental
```
Then, there are three sub-stages for training (5-shot 16+3 setting):
- Sub-Stage1: training close set module
```
python -u main.py --output_dir ./output_stage1_16 --gpu_id 0,1
```
- Sub-Stage2: training meta channel module
```
python -u main.py --finetune --ckpt ./output_stage1_16/final.pth --output_dir ./output_stage2_16/ --total_itrs 10000 --gpu_id 0,1
```
- Sub-Stage3: training region-aware metric learning module
```
python -u main_metric.py --ckpt ./output_stage2_16/final.pth --output_dir ./output_stage3_16/  --novel_dir ./novel/
```

### Inference
For 16+3 5-shots:
```
python main_metric.py --ckpt ./output_stage3_16/final.pth --test_only --test_mode 16_3  --novel_dir ./novel
```
For 16+1 5-shots:
```
python main_metric.py --ckpt ./output_stage3_16/final.pth --test_only --test_mode 16_3  --novel_dir ./novel_1
```
For 16+1 5-shots:
```
python main_metric.py --ckpt ./output_stage3_16/final.pth --test_only --test_mode 16_1  --novel_dir ./novel
```
For 12+7 5-shots:
```
python main_metric.py --ckpt ./output_stage3_12/final.pth --test_only --test_mode 12  --novel_dir ./novel
```

## Citation
```
@inproceedings{raml2022,
author = {Dong, Hexin and Chen, Zifan and Yuan, Mingze and Xie, Yutong and Zhao, Jie and Yu, Fei and Dong, Bin and Zhang, Li},
title = {Region-Aware Metric Learning for Open World Semantic Segmentation via Meta-Channel Aggregation},
booktitle = {31th International Joint Conference on Artificial Intelligence (IJCAI-22)},
year = {2022},
}
```
