# Pytorch Implementation of Point Transformers

Pytorch Implementation of Point Transformers with HAQ Automated Quantization，

基于Point Transformers复现点云分割任务，并使用HAQ算法进行自动量化（2bit和4bit）压缩，几乎不影响精度

## 准备数据：
使用连接下载 **ShapeNet** 数据集：[下载地址](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) 

下载完成后解压到 `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

## 预训练：

```bash
bash run/pretrain.sh
```

## 强化学习搜索：

```bash
bash run/search.sh
```

## 量化后微调：

```bash
bash run/finutune.sh
```

## 实验结果：


| Models                   | Accuracy | cat.mIOU | ins.mIOU |
| ------------------------ | -------------- | ------------ | ------------ |
| Point Transformer (paper)      |       None      |     0.837    |    0.866     |
| Point Transformer (our-no quant)  |       0.93535      |     0.79958    |    0.83802     |
| Point Transformer (our-0.5×preserve)  |       not yet      |     not yet    |    not yet     |
