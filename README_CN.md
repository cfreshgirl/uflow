# 目录

<!-- TOC -->

- [目录](#目录)
- [uflow描述](#uflow描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出MindIR)
        - [在Ascend310执行推理](#在Ascend310执行推理)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ImageNet上的HarDNet](#ImageNet上的hardnet)
        - [推理性能](#推理性能)
            - [ImageNet上的HarDNet](#ImageNet上的hardnet)
    - [使用流程](#使用流程)
        - [推理](#推理)
        - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# uflow描述

光流是计算机视觉中描述两幅图像之间像素级对应关系的关键表示方法。我们系统地比较和分析了无监督光流中的一组关键组件，以确定哪些光度损失、遮挡处理和光滑正则化是最有效的。在此基础上，我们对无监督流模型进行了一些新的改进，如成本体积归一化、停止在遮挡掩模处的梯度、在流场上采样前鼓励平滑，以及调整图像大小的持续自我监督。通过将我们的调查结果与我们改进的模型组件相结合，我们能够提出一种新的无监督流技术，它显著优于以前的无监督光流。

[论文](https://paperswithcode.com/paper/what-matters-in-unsupervised-optical-flow)：Jonschkowski R ,  Stone A ,  Barron J T , et al. What Matters in Unsupervised Optical Flow[C]// 2020.

# 模型架构

估计光流的任务可以定义为：给定两个颜色图像I(1)，I(2)∈RH×W×3，我们想估计流场V (1)∈RH×W×2，对于每个像素I(1)表示对应的像素的相对位置在I(2)。请注意，光流是像素运动的非对称表示： V (1)为I (1)中的每个像素提供了一个流向量，但是要找到从图像2到图像1的映射，就需要估计V (2)。我们将首先讨论一个模型fθ(·)，基于PWC-Net ，并通过成本-体积归一化进行改进。

# 数据集

使用的数据集：

- [flyingchairs](http://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip)
  “飞椅”是一个具有光流地面真实性的合成数据集。它由22872个图像对和相应的流场组成。图像显示了随机背景和移动的3D椅子模型，椅子和背景都会映射到平面图像中。飞椅中图片是.ppm格式，光流是.flo格式。所有的文件都在data文件夹下。
- [sintel](http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip)
  MPI Sintel数据集是用于训练和评估光流算法的最广泛使用的数据集之一。该数据集包含了一系列额外的挑战，如远距离运动、光照变化、镜面反射、运动模糊和大气效应。它是第一个实现广泛使用的合成数据集，因为它很好地代表了自然场景和运动特征。训练集由1040个地面真实光流组成，测试集包含12个序列的564张图片，平均速度和最大速度分别为5和445。

数据预处理：

```bash
#flyingchairs数据集(将zip文件解压)
用法：python -m src.utils.convert_flying_chairs_to_mindrecords --data_dir "" --output_dir "" --shard=0 --num_shards=1
实例：python -m src.utils.convert_flying_chairs_to_mindrecords --data_dir='/path/FlyingChairs_release/data' --output_dir='/path/flyingchairs' --shard=0 --num_shards=1

#sintel数据集(将下载的zip文件解压到Sintel文件夹下)
用法：python -m src.utils.convert_sintel_to_mindrecords --data_dir "" --output_dir "" --shard=0 --num_shards=1
实例：python -m src.utils.convert_sintel_to_mindrecords --data_dir='/path/Sintel/' --output_dir='/path/sintel' --shard=0 --num_shards=1
```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/others/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend或GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```bash
  #1P训练
  #flyingchairs数据集
  用法：python -m train --train_on "" --height "" --width "" --lr "" --checkpoint_dir "" > train.log 2>&1 &
  实例：python -m train --train_on="flyingchairs:/path/flyingchairs/train/" --height=384 --width=512 --lr 0.0001 --checkpoint_dir='/path/checkpoints_flyingchairs/uflow'> train.log 2>&1 &
  or
  用法：bash run_single_train.sh [DATASET_PATH] [HEIGHT] [WIDTH] [LR] [CKPT_DIR] [DEVICE_ID]
  实例：bash run_single_train.sh flyingchairs:/path/flyingchairs/train/ 384 512 0.0001 /path/checkpoints_flyingchairs/ 0

  #sintel数据集:
  用法：python -m train --train_on "" --height "" --width "" --lr "" --checkpoint_dir "" --pre_trained True --pre_ckpt_path "" > train.log 2>&1 &
  实例：python -m train --train_on="sintel-clean:/path/sintel/test/clean"  --height=448 --width=1024 --lr 0.00005 --checkpoint_dir='/path/checkpoints_sintel_clean/uflow/' --pre_trained=True --pre_ckpt_path='/path/pretrain.ckpt' > train.log 2>&1 &
  or
  用法：bash run_single_train.sh [DATASET_PATH] [HEIGHT] [WIDTH] [LR] [CKPT_DIR] [PRE_TRAINED] [PRE_CKPT_PATH] [DEVICE_ID]
  实例：bash run_single_train.sh sintel-clean:/path/sintel/test/clean  448 1024 0.00005 /path/checkpoints_sintel_clean/uflow/ True /path/pretrain.ckpt 0

  # 运行分布式训练示例
  #flyingchairs数据集
  用法：bash run_distribute_train.sh [DATASET_PATH] [HEIGHT] [WIDTH] [LR] [CKPT_DIR]
  实例：bash run_distribute_train.sh flyingchairs:/path/flyingchairs/train/ 384 512 0.0001 /path/checkpoints_flyingchairs/

  #sintel数据集:
  用法：bash run_distribute_train.sh [DATASET_PATH] [HEIGHT] [WIDTH] [LR] [CKPT_DIR] [PRE_TRAINED] [PRE_CKPT_PATH]
  实例：bash run_distribute_train.sh sintel-clean:/path/sintel/test/clean  448 1024 0.00005 /path/checkpoints_sintel_clean/uflow/ True /path/pretrain.ckpt

  # 运行评估示例
  #flyingchairs数据集
  用法：python -m eval --eval_on "" --height "" --width "" --checkpoint_dir "" > eval.log 2>&1 &
  实例：python -m eval --eval_on="flyingchairs:/path/flyingchairs/test/" --height=384 --width=512 --checkpoint_dir='/path/checkpoints_flyingchairs/uflow'> eval_flyingchairs.log 2>&1 &
  or
  用法：bash run_eval.sh [DATASET_PATH] [HEIGHT] [WIDTH] [CKPT_DIR] [DEVICE_ID]
  实例：bash run_eval.sh flyingchairs:/path/flyingchairs/test/ 384 512 /path/checkpoints_flyingchairs/ 0

  #sintel数据集:
  用法：python -m eval --eval_on "" --height "" --width "" --checkpoint_dir "" --pre_trained True --pre_ckpt_path "" > eval.log 2>&1 &
  实例：python -m eval --eval_on="sintel-clean:/path/sintel/training/clean"  --height=448 --width=1024 --checkpoint_dir='/path/checkpoints_sintel_clean/uflow/' > eval_clean.log 2>&1 &
  or
  用法：bash run_eval.sh [DATASET_PATH] [HEIGHT] [WIDTH] [CKPT_DIR] [DEVICE_ID]
  实例：bash run_eval.sh sintel-clean:/path/sintel/training/clean  448 1024 /path/checkpoints_sintel_clean/uflow/ 0
  ```

# 脚本说明

## 脚本及样例代码

```bash
└── uflow
    ├── README_CN.md                    // uflow相关说明
    ├── scripts
    │   ├── run_single_train.sh             // 单卡到Ascend的shell脚本
    │   ├── run_distribute_train.sh             // 分布式到Ascend的shell脚本
    │   └── run_eval.sh              // Ascend评估的shell脚本
    ├── src
        ├── utils
            ├── conversion_utils.py     //数据预处理转换
            ├── convert_flying_chairs_to_mindrecords.py     //将flyingchairs数据集转换为mindrecord格式
            ├── convert_sintel_to_mindrecords.py      //将sintel数据集转换为mindrecord格式
            ├── uflow_resampler.py      //数据采样
            └── uflow_utils.py      //求loss中间计算
        ├── dataset.py             // 创建数据集
        ├── Uflow.py               // hardnet架构
        ├── network_with_loss.py        // loss函数
        ├── config.py                 //参数配置
        ├── infer.py             //评估网络
        ├── logger.py          //日志打印
        └── uflow_augmentation.py          //数据增强
    ├── train.py               // 训练脚本
    ├── eval.py               // 评估脚本
    └── export.py             //将checkpoint文件导出为air/mindair文件
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

  ```python
  "batch_size": 1                                                     #训练批次大小
  "learning_rate": 0.0001                                             #学习率
  "smoothness_edge_weighting": "exponential"                          #平滑边缘权重
  "weight_census": 1.0                                                #loss中census权重系数
  "weight_smooth1": 4.0                                               #loss中smooth1权重系数
  "weight_smooth2": 0.0                                               #loss中smooth2权重系数
  "smoothness_edge_constant": 150.                                    #平滑边缘常量
  "weight_selfsup": 0.6                                               #自我监督权重系数
  "epoch_size": 1000                                                  #总计训练epoch数
  "epoch_length": 1000                                                #每个epoch的step数
  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  #flyingchairs数据集
  用法：python -m train --train_on "" --height "" --width "" --lr "" --checkpoint_dir "" > train.log 2>&1 &
  实例：python -m train --train_on="flyingchairs:/path/flyingchairs/train/" --height=384 --width=512 --lr 0.0001 --checkpoint_dir='/path/checkpoints_flyingchairs/uflow'> train.log 2>&1 &
  or
  用法：bash run_single_train.sh [DATASET_PATH] [HEIGHT] [WIDTH] [LR] [CKPT_DIR] [DEVICE_ID]
  实例：bash run_single_train.sh flyingchairs:/path/flyingchairs/train/ 384 512 0.0001 /path/checkpoints_flyingchairs/ 0

  #sintel数据集:
  用法：python -m train --train_on "" --height "" --width "" --lr "" --checkpoint_dir "" --pre_trained True --pre_ckpt_path "" > train.log 2>&1 &
  实例：python -m train --train_on="sintel-clean:/path/sintel/test/clean"  --height=448 --width=1024 --lr 0.00005 --checkpoint_dir='/path/checkpoints_sintel_clean/uflow/' --pre_trained=True --pre_ckpt_path='/path/pretrain.ckpt' > train.log 2>&1 &
  or
  用法：bash run_single_train.sh [DATASET_PATH] [HEIGHT] [WIDTH] [LR] [CKPT_DIR] [PRE_TRAINED] [PRE_CKPT_PATH] [DEVICE_ID]
  实例：bash run_single_train.sh sintel-clean:/path/sintel/test/clean  448 1024 0.00005 /path/checkpoints_sintel_clean/uflow/ True /path/pretrain.ckpt 0
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # grep "loss is " train.log
  epoch[1], loss3.4317467,  lr:0.0001, per step time: 1797.258ms
  epoch[2], loss3.2129548,  lr:0.0001, per step time: 1832.040ms
  epoch[3], loss3.1161423,  lr:0.0001, per step time: 189.984ms
  ...
  ```

  模型检查点保存在当前目录下。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  #flyingchairs数据集
  用法：bash run_distribute_train.sh [DATASET_PATH] [HEIGHT] [WIDTH] [LR] [CKPT_DIR]
  实例：bash run_distribute_train.sh flyingchairs:/path/flyingchairs/train/ 384 512 0.0001 /path/checkpoints_flyingchairs/

  #sintel数据集:
  用法：bash run_distribute_train.sh [DATASET_PATH] [HEIGHT] [WIDTH] [LR] [CKPT_DIR] [PRE_TRAINED] [PRE_CKPT_PATH]
  实例：bash run_distribute_train.sh sintel-clean:/path/sintel/test/clean  448 1024 0.00005 /path/checkpoints_sintel_clean/uflow/ True /path/pretrain.ckpt
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train_parallel[X]/log文件查看结果。采用以下方式达到损失值：

  ```bash
  # grep "result:" device*/log
  device0/log:epoch:1 epoch 1, loss2.6042662,  per step time: 1478.071321725ms
  device0/log:epcoh:2 epoch 2, loss2.6292048,  per step time: 1477.836974382ms
  ...
  device1/log:epoch:1 epoch 1 , loss2.6231596,  per step time: 1192.178252458ms
  device1/log:epcoh:2 epoch 2 , loss2.517363,  per step time: 1192.181766271ms
  ...
  ...
  ```

## 评估过程

### 评估

- 在Ascend环境运行评估

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/hardnet/train_hardnet_390.ckpt”。

  ```bash
  #flyingchairs数据集
  用法：python -m eval --eval_on "" --height "" --width "" --checkpoint_dir "" > eval.log 2>&1 &
  实例：python -m eval --eval_on="flyingchairs:/path/flyingchairs/test/" --height=384 --width=512 --checkpoint_dir='/path/checkpoints_flyingchairs/uflow'> eval_flyingchairs.log 2>&1 &
  OR
  用法：bash run_eval.sh [DATASET_PATH] [HEIGHT] [WIDTH] [CKPT_PATH] [DEVICE_ID]
  实例：bash run_eval.sh flyingchairs:/path/flyingchairs/test/ 384 512 /path/checkpoints_flyingchairs/uflow_1200_1000.ckpt 0

  #sintel数据集:
  用法：python -m eval --eval_on "" --height "" --width "" --checkpoint_dir "" --pre_trained True --pre_ckpt_path "" > eval.log 2>&1 &
  实例：python -m eval --eval_on="sintel-clean:/path/sintel/training/clean"  --height=448 --width=1024 --checkpoint_dir='/path/checkpoints_sintel_clean/uflow/' > eval_clean.log 2>&1 &
  OR
  用法：bash run_eval.sh [DATASET_PATH] [HEIGHT] [WIDTH] [CKPT_PATH] [DEVICE_ID]
  实例：bash run_eval.sh sintel-clean:/path/sintel/training/clean  448 1024 /path/checkpoints_sintel_clean/uflow_1200_1000.ckpt 0

  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```text
  #flyingchairs数据集
  'flyingchairsEPE': 2.735035
```

```text
  #sintel-clean数据集
  'sintel-cleanEPE': 3.345035
```

## 推理过程

### 导出MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --height [HEIGHT] --width [WIDTH] --file_format [FILE_FORMAT]
```

- `ckpt_file` ckpt文件路径
- `height` 数据的height
- `width` 数据的width
- `file_format` 导出模型格式，["AIR", "ONNX", "MINDIR"]

### 在Ascend310执行推理

暂不支持，待补充

# 模型描述

## 性能

### 评估性能

#### flyingchairs上的uflow

| 参数                 | Ascend 910
| -------------------------- | -------------------------------------- |
| 模型版本              | uflow
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期              | 2022-10-15 |
| MindSpore版本          | r1.8 |
| 数据集                    | flyingchairs |
| 训练参数        | epoch=1200, steps per epoch=1000, batch_size = 1 |
| 优化器                  | Adam |
| 损失函数              | 自定义 |
| 输出                    | EPE |
| 损失                       | 2.0674517 |
| 速度                      | 294.862毫秒/步（8卡）|
| 总时长                 | 85小时 |
| 参数(M)             | 210M |
| 微调检查点| 89M（.ckpt文件）|

#### sintel上的uflow

| 参数                 | Ascend 910
| -------------------------- | -------------------------------------- |
| 模型版本              | uflow
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期              | 2022-10-15 |
| MindSpore版本          | r1.8 |
| 数据集                    | sintel |
| 训练参数        | epoch=1200, steps per epoch=1000, batch_size = 1 |
| 优化器                  | Adam |
| 损失函数              | 自定义 |
| 输出                    | EPE |
| 损失                       | 2.7350647 |
| 速度                      | 294.862毫秒/步（8卡）|
| 总时长                 | 85小时 |
| 参数(M)             | 210M |
| 微调检查点| 89M（.ckpt文件）|

### 迁移学习

待补充

# 随机情况说明

无

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
