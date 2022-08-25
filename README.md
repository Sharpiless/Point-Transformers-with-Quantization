# Pytorch Implementation of Point Transformers



基于Point Transformers复现点云分割任务，并使用HAQ算法进行自动量化（2bit和4bit）压缩，几乎不影响精度

## 准备数据：
使用链接下载 **ShapeNet** 数据集：[下载地址](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) 


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
## 解决问题记录：

- 梯度更新不一致问题：原因是每次根据loss更新参数时梯度没有清零，使用的是累计梯度，添加'self.optimizer.zero_grad()'即可
- Acc等指标计算错误问题：在计算mIOU时开始是使用一个batch的数据求mIOU再最后取平均，这样一个batch某些类数据量可能为0导致计算有偏差，改成最后一起求mIOU即可
- 模型量化后Acc不变的问题：这个问题最难解决，最后发现是transform里面linear往往参数较少，使用kmeans聚类算法（指定聚类中心数目）导致某些聚类中心没有数据，对应的mask产生0值；在使用这些mask更新参数时则会导致模型参数更新为nan，输出nan，使得参数不再更新，模型输出每次都完全相同

## 论文复现结果：

### ShapeNet：

| Class               | mIoU     | 
| ------------------- | -------- |
|  Airplane  |  0.7901    | 
|  Bag  |  0.7901    | 
|  Cap  |  0.8042   | 
|  Car  |  0.8287    | 
|  Chair  |  0.8985    | 
|  Earphone  |  0.7293    | 
|  Guitar  |  0.8979    | 
|  Knife  |  0.8654    | 
|  Lamp  |  0.8211    | 
|  Laptop  |  0.9524    | 
|  Motorbike  |  0.5616    | 
|  Mug  |  0.9288    | 
|  Pistol  |  0.7693    | 
|  Rocket  |  0.5708    | 
|  Skateboard  |  0.7270    | 
|  Table  |  0.8190    | 
|  Total  |  0.7940    | 

| ShapeNet                   | Accuracy | cat.mIOU | ins.mIOU |
| ------------------------ | -------------- | ------------ | ------------ |
| Point Transformer (papers)      |       None      |     0.837    |    0.866     |
| Point Transformer (ours)  |       0.93535      |     0.79958    |    0.83802     |

### S3DIS：

| S3DIS                   | Accuracy |
| ------------------------ | -------------- | 
| Point Transformer (papers)      |       0.908      | 
| Point Transformer (ours)  |       0.846      |  

trained only for 4 epoches due to hardware limitation.



## 针对量化后准确率降低的改进方案：

- 对于模型前几层和最后的分类（分割）层对准确率影响较大，因此使用较高位数量化或者不进行量化
- 使用知识蒸馏的方法，在保证模型大小不变的情况下提升准确率

## 量化实验结果(ShapeNet)：


| Models                   | Accuracy | cat.mIOU | ins.mIOU |
| ------------------------ | -------------- | ------------ | ------------ |
| Point Transformer (paper)      |       None      |     0.837    |    0.866     |
| Point Transformer (our-no quant)  |       0.93535      |     0.79958    |    0.83802     |
| Point Transformer (our-no quant, mix)  |       0.93653      |     0.791004    |   0.838491     |
| Point Transformer (our-0.1×preserve)  |      0.932     |     0.781    |    0.826     |
| Point Transformer (our-0.1×preserve, mix)  |      0.9341     |     0.7894    |    0.8337     |
| Point Transformer (our-0.1×preserve, mix, finetune)  |      0.936523     |     0.796603    |    0.837771     |
| Point Transformer (our-0.1×preserve, mix, finetune, distill)  |      0.940213     |     0.799304    |    0.839487     |

# 2.2 Pytorch Implementation of Point Transformers

基于PCT: Point Cloud Transformer复现点云分割任务，还未进行量化工作。
## 运行：
```bash
PCT.ipynb
```
## 论文复现结果：
### modelnet40(PCT)：

| modelnet40                   | Accuracy |
| ------------------------ | -------------- | 
| PCT (papers)      |       0.932      | 
| PCT (ours)  |       0.896576      |  
