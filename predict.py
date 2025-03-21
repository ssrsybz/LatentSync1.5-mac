# Prediction interface for Cog ⚙️
# https://cog.run/python

"""
Cog预测接口模块

本模块实现基于Replicate平台的模型预测接口，包含：
1. 模型权重下载与缓存管理
2. 辅助模型的符号链接创建
3. 视频推理流程执行
"""

from cog import BasePredictor, Input, Path
import os
import time
import subprocess

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/chunyu-li/LatentSync/model.tar"


def download_weights(url, dest):
    """
    模型权重下载函数
    
    参数：
        url: 模型权重下载URL
        dest: 本地存储路径
    
    使用pget工具实现大文件断点续传下载
    """
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    """
    Cog预测器主类
    
    职责：
    - 初始化阶段加载模型权重
    - 创建必要的符号链接
    - 执行视频推理流程
    """
    def setup(self) -> None:
        """
        初始化模型环境
        
        执行步骤：
        1. 检查并下载主模型权重
        2. 创建辅助模型的符号链接到Torch缓存目录
        3. 准备模型推理所需目录结构
        """
        """Load the model into memory to make running multiple predictions efficient"""
        # Download the model weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Soft links for the auxiliary models
        os.system("mkdir -p ~/.cache/torch/hub/checkpoints")
        os.system(
            "ln -s $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip"
        )
        os.system(
            "ln -s $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth"
        )
        os.system(
            "ln -s $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth"
        )

    def predict(
        self,
        video: Path = Input(description="输入视频文件", default=None),
        audio: Path = Input(description="输入音频文件", default=None),
        guidance_scale: float = Input(description="引导系数，范围[1,2.5]", ge=1, le=2.5, default=1.5),
        seed: int = Input(description="随机种子，0表示随机生成", default=0),
    ) -> Path:
        """
        执行视频推理流程
        
        参数处理逻辑：
        - 自动生成随机种子（当seed<=0时）
        - 构建推理命令行参数
        - 调用scripts.inference执行核心推理逻辑
        
        返回：生成视频的临时文件路径
        """
        """Run a single prediction on the model"""
        if seed <= 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        video_path = str(video)
        audio_path = str(audio)
        config_path = "configs/unet/stage2.yaml"
        ckpt_path = "checkpoints/latentsync_unet.pt"
        output_path = "/tmp/video_out.mp4"

        # 构建并执行推理命令，包含以下关键参数：
# --unet_config_path: UNet模型配置路径
# --inference_ckpt_path: 训练好的模型权重路径
# --guidance_scale: 引导系数控制生成效果
# --video_path/--audio_path: 输入媒体文件路径
# --seed: 随机种子保证可重复性
        # 检查是否支持MPS设备
        try:
            import torch
            device_arg = "" if not torch.backends.mps.is_available() else "--device mps"
        except ImportError:
            device_arg = ""
            
        # 使用subprocess.run替代os.system以获得更好的错误处理
        try:
            cmd = f"python -m scripts.inference --unet_config_path {config_path} --inference_ckpt_path {ckpt_path} --guidance_scale {str(guidance_scale)} --video_path {video_path} --audio_path {audio_path} --video_out_path {output_path} --seed {seed} {device_arg}"
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"推理过程失败：{str(e)}")

        return Path(output_path)
