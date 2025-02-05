## Environment

- Python 3.11.4
- Anaconda3 2023.07-2
- conda 23.7.4
- cuda 12.1
- pandas 2.0.3
- numpy 1.23.5
- pytorch 2.1.0
- torchsummary 1.5.1
- torchvision 0.16.0
- matplotlib 3.7.2

## Introduction

- 共有 3 个代码文件:
  - 模型架构 model.py
  - 模型训练 model_train.py
  - 模型测试 model_t.py
- 数据记录文件：
  - 使用 CIFAR10 数据集的记录文件：CIFAR10 记录
  - 使用 MNIST 数据集的记录文件：MNIST 记录
- 实验报告

## Steps

- 首先进入项目文件夹…/LeNet，然后运行 model_train.py 文件`python model_train.py`，开始训练。
- 其中，调整 model_train.py 文件代码第 44-51 行，以改变优化方法（动量 SGD、AdaDelta、RMSprop 和 Adam）
- 每个文件由两部分组成，前一部分为使用 MNIST 数据集的代码，后一部分为使用 CIFAR10 数据集的代码，
  - 若想使用 MNIST 数据集进行实验，则注释掉每个文件的后半部分即可（使用 CIFAR10 数据集的代码）；
  - 反之则注释掉前半部分

## Notes

- 加载最高准确率下的模型参数时，需根据每台设备的相关文件路径存放最优模型参数的路径，需存放到./LeNet 目录下
  - 如 `torch.save(best_model_wts, 'D:/pythonProject/lesson3A/LeNet/best_model.pth')`

## Dataset

- MNIST 手写数字
- 通过 `from torchvision.datasets import MNIST` 导入数据集
- 将数据集按比例划分成训练数据集和验证数据集
