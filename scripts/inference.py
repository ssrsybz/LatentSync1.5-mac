# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 ("许可证");
# 您除非符合许可证条件，否则不得使用本文件
# 您可以在以下网址获取许可证副本：
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，按「原样」分发的软件
# 没有任何明示或暗示的担保或条件
# 请参阅许可证了解特定语言的管理权限和限制

import argparse
import os
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature


def main(config, args):
    # 参数校验：验证输入视频和音频文件是否存在
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"视频路径 '{args.video_path}' 不存在")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"音频路径 '{args.audio_path}' 不存在")

    # 自动选择计算设备和计算精度
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    is_fp16_supported = (device == "cuda" and torch.cuda.get_device_capability()[0] > 7) or device == "mps"
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"输入视频路径: {args.video_path}")
    print(f"输入音频路径: {args.audio_path}")
    print(f"加载的检查点路径: {args.inference_ckpt_path}")

    # 初始化扩散模型调度器
    scheduler = DDIMScheduler.from_pretrained("configs")

    # 根据UNet的交叉注意力维度选择对应的Whisper模型
    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"  # 大尺寸语音模型
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"   # 小尺寸语音模型
    else:
        raise NotImplementedError("交叉注意力维度必须是768或384")

    # 初始化音频特征编码器
    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device=device,
        num_frames=config.data.num_frames,          # 视频帧数配置
        audio_feat_length=config.data.audio_feat_length,  # 音频特征长度
    )

    # 初始化VAE并配置潜在空间缩放参数
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215  # SD VAE的标准缩放系数
    vae.config.shift_factor = 0           # 无偏移调整

    # 加载预训练的UNet模型
    denoising_unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),  # 从配置转换模型参数
        args.inference_ckpt_path,              # 检查点路径
        device="cpu",                          # 初始加载到CPU
    )
    denoising_unet = denoising_unet.to(device=device, dtype=dtype)

    # 构建完整的唇形同步推理管道
    pipeline = LipsyncPipeline(
        vae=vae,                    # 变分自编码器
        audio_encoder=audio_encoder, # 音频特征编码器
        denoising_unet=denoising_unet, # 去噪UNet
        scheduler=scheduler,        # 扩散调度器
    ).to(device)                    # 将整个管道移至指定设备

    # 设置随机种子保证可重复性
    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()  # 使用随机种子
    print(f"初始种子: {torch.initial_seed()}")

    # 执行核心推理流程
    pipeline(
        video_path=args.video_path,    # 输入视频路径
        audio_path=args.audio_path,    # 输入音频路径
        video_out_path=args.video_out_path,  # 输出视频路径
        video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),  # 遮罩路径
        num_frames=config.data.num_frames,        # 处理帧数
        num_inference_steps=args.inference_steps, # 扩散步数
        guidance_scale=args.guidance_scale,       # 分类器自由引导系数
        weight_dtype=dtype,            # 计算精度
        width=config.data.resolution,  # 输出分辨率宽度
        height=config.data.resolution,  # 输出分辨率高度
        mask_image_path=config.data.mask_image_path,  # 面部遮罩图像路径
    )


if __name__ == "__main__":
    # 配置命令行参数解析器
    parser = argparse.ArgumentParser(description='唇形同步推理脚本')
    
    # 模型配置参数
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml",
                        help='UNet模型配置文件路径')
    parser.add_argument("--inference_ckpt_path", type=str, required=True,
                        help='推理使用的模型检查点路径')
    
    # 输入输出参数
    parser.add_argument("--video_path", type=str, required=True,
                        help='输入视频文件路径')
    parser.add_argument("--audio_path", type=str, required=True,
                        help='输入音频文件路径')
    parser.add_argument("--video_out_path", type=str, required=True,
                        help='输出视频文件保存路径')
    
    # 推理过程参数
    parser.add_argument("--inference_steps", type=int, default=20,
                        help='扩散过程的推理步数（默认：20）')
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help='分类器自由引导系数（默认：1.0）')
    parser.add_argument("--seed", type=int, default=1247,
                        help='随机种子（-1表示随机生成）')
    
    args = parser.parse_args()
    
    # 加载UNet配置
    config = OmegaConf.load(args.unet_config_path)
    
    main(config, args)
