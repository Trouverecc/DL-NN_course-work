# Usage example

`python train_cifar10.py --net vit_small --n_epochs 400` # vit-small

`python train_cifar10.py --net swin --n_epochs 400` # train with SwinTransformers

`python train_cifar10.py --net swin-revised --n_epochs 400` # train with swin-revised

# Model Export

This repository supports exporting trained models to ONNX and TorchScript formats for deployment purposes. You can export your trained models using the `export_models.py` script.

### Basic Usage

```bash
python export_models.py --checkpoint path/to/checkpoint --model_type vit --output_dir exported_models
```

# Tip

代码参考 Kentaro Yoshioka 进行修改，进行基于 Swin Transformer 的调整（文件名称：swin-revised.py），引入混合注意力机制和轻量化设计思想。

```
@misc{yoshioka2024visiontransformers,
  author       = {Kentaro Yoshioka},
  title        = {vision-transformers-cifar10: Training Vision Transformers (ViT) and related models on CIFAR-10},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/kentaroy47/vision-transformers-cifar10}}
}
```
