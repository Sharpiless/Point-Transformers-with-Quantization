# Pytorch Implementation of Point Transformers

Pytorch Implementation of Point Transformers with HAQ Automated Quantization，

基于Point Transformers复现点云分割任务，并使用HAQ算法进行自动量化压缩，几乎不影响精度

### Data Preparation
Download alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`.

### Pretrain
Change which method to use in `config/partseg.yaml` and run
```
python train_partseg.py
```
### Results

Net yet.

