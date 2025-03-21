#!/bin/bash

# 创建新的conda环境（Python 3.10.13）
conda create -y -n latentsync python=3.10.13  # -y参数自动确认操作
conda activate latentsync  # 激活新创建的环境

# 安装ffmpeg多媒体处理工具
conda install -y -c conda-forge ffmpeg  # 使用conda-forge渠道安装

# 安装Python依赖包
pip install -r requirements.txt  # 根据requirements.txt安装所有依赖

# 安装OpenCV系统依赖（Linux/Mac适配）
# Mac用户请使用：brew install libglvnd 并注释下一行
sudo apt -y install libgl1  # Linux系统下安装OpenCV的图形库依赖

# 从HuggingFace下载所有预训练模型
huggingface-cli download ByteDance/LatentSync-1.5 --local-dir checkpoints \
  --exclude "*.git*" "README.md"  # 排除git相关文件和说明文档

# 为辅助模型创建软链接（解决路径引用问题）
mkdir -p ~/.cache/torch/hub/checkpoints  # 确保目标目录存在
# 创建模型文件的符号链接到标准缓存目录
ln -s $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip  # 人脸对齐模型
ln -s $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth  # 人脸检测模型
ln -s $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth  # 图像特征提取模型
