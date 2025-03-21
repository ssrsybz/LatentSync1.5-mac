# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入必要的库和模块
import os
import math
import argparse
import shutil
import datetime
import logging
from omegaconf import OmegaConf  # 用于处理配置文件

from tqdm.auto import tqdm  # 进度条显示
from einops import rearrange  # 张量维度重排

# PyTorch相关导入
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist  # 分布式训练
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# diffusers库相关导入
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler  # VAE和DDIM采样器
from diffusers.utils.logging import get_logger
from diffusers.optimization import get_scheduler
from accelerate.utils import set_seed

# 自定义模块导入
from latentsync.data.unet_dataset import UNetDataset  # 数据集类
from latentsync.models.unet import UNet3DConditionModel  # UNet模型
from latentsync.models.stable_syncnet import StableSyncNet  # SyncNet模型
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline  # 唇形同步pipeline
from latentsync.utils.util import (
    init_dist,  # 初始化分布式训练
    cosine_loss,  # 余弦损失函数
    one_step_sampling,  # 单步采样
)
from latentsync.utils.util import plot_loss_chart  # 绘制损失曲线
from latentsync.whisper.audio2feature import Audio2Feature  # 音频特征提取
from latentsync.trepa.loss import TREPALoss  # TREPA损失
from eval.syncnet import SyncNetEval  # SyncNet评估
from eval.syncnet_detect import SyncNetDetector  # SyncNet检测器
from eval.eval_sync_conf import syncnet_eval  # SyncNet评估函数
import lpips  # 感知损失


logger = get_logger(__name__)


def main(config):
    # 初始化分布式训练环境
    local_rank = init_dist()
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    # 设置随机种子
    seed = config.run.seed + global_rank
    set_seed(seed)

    # 创建输出目录
    folder_name = "train" + datetime.datetime.now().strftime(f"-%Y_%m_%d-%H:%M:%S")
    output_dir = os.path.join(config.data.train_output_dir, folder_name)

    # 配置日志格式
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # 在主进程中创建输出目录和复制配置文件
    if is_main_process:
        diffusers.utils.logging.set_verbosity_info()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/val_videos", exist_ok=True)
        os.makedirs(f"{output_dir}/sync_conf_results", exist_ok=True)
        shutil.copy(config.unet_config_path, output_dir)
        shutil.copy(config.data.syncnet_config_path, output_dir)

    device = torch.device(local_rank)

    # 初始化DDIM采样器
    noise_scheduler = DDIMScheduler.from_pretrained("configs")

    # 加载预训练的VAE模型
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae.requires_grad_(False)  # 冻结VAE参数
    vae.to(device)

    if config.run.pixel_space_supervise:
        vae.enable_gradient_checkpointing()

    # 加载SyncNet评估模型
    syncnet_eval_model = SyncNetEval(device=device)
    syncnet_eval_model.loadParameters("checkpoints/auxiliary/syncnet_v2.model")

    syncnet_detector = SyncNetDetector(device=device, detect_results_dir="detect_results")

    # 根据cross_attention_dim选择Whisper模型大小
    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    # 初始化音频特征提取器
    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device=device,
        audio_embeds_cache_dir=config.data.audio_embeds_cache_dir,
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    # 加载UNet模型
    denoising_unet, resume_global_step = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        config.ckpt.resume_ckpt_path,
        device=device,
    )

    # 如果使用音频层和SyncNet
    if config.model.add_audio_layer and config.run.use_syncnet:
        syncnet_config = OmegaConf.load(config.data.syncnet_config_path)
        if syncnet_config.ckpt.inference_ckpt_path == "":
            raise ValueError("SyncNet path is not provided")
        syncnet = StableSyncNet(OmegaConf.to_container(syncnet_config.model), gradient_checkpointing=True).to(
            device=device, dtype=torch.float16
        )
        syncnet_checkpoint = torch.load(
            syncnet_config.ckpt.inference_ckpt_path, map_location=device, weights_only=True
        )
        syncnet.load_state_dict(syncnet_checkpoint["state_dict"])
        syncnet.requires_grad_(False)  # 冻结SyncNet参数

        del syncnet_checkpoint
        torch.cuda.empty_cache()

    # 设置可训练参数
    if config.model.use_motion_module:
        denoising_unet.requires_grad_(False)
        for name, param in denoising_unet.named_parameters():
            for trainable_module_name in config.run.trainable_modules:
                if trainable_module_name in name:
                    param.requires_grad = True
                    break
        trainable_params = list(filter(lambda p: p.requires_grad, denoising_unet.parameters()))
    else:
        denoising_unet.requires_grad_(True)
        trainable_params = list(denoising_unet.parameters())

    # 根据进程数调整学习率
    if config.optimizer.scale_lr:
        config.optimizer.lr = config.optimizer.lr * num_processes

    # 初始化优化器
    optimizer = torch.optim.AdamW(trainable_params, lr=config.optimizer.lr)

    if is_main_process:
        logger.info(f"trainable params number: {len(trainable_params)}")
        logger.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # 启用梯度检查点
    if config.run.enable_gradient_checkpointing:
        denoising_unet.enable_gradient_checkpointing()

    # 创建训练数据集和数据加载器
    train_dataset = UNetDataset(config.data.train_data_dir, config)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=config.run.seed,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=config.data.num_workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn,
    )

    # 计算训练步数
    if config.run.max_train_steps == -1:
        assert config.run.max_train_epochs != -1
        config.run.max_train_steps = config.run.max_train_epochs * len(train_dataloader)

    # 初始化学习率调度器
    lr_scheduler = get_scheduler(
        config.optimizer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.lr_warmup_steps,
        num_training_steps=config.run.max_train_steps,
    )

    # 初始化感知损失函数
    if config.run.perceptual_loss_weight != 0 and config.run.pixel_space_supervise:
        lpips_loss_func = lpips.LPIPS(net="vgg").to(device)

    # 初始化TREPA损失函数
    if config.run.trepa_loss_weight != 0 and config.run.pixel_space_supervise:
        trepa_loss_func = TREPALoss(device=device, with_cp=True)

    # 创建验证pipeline
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        denoising_unet=denoising_unet,
        scheduler=noise_scheduler,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    # 使用DDP包装模型
    denoising_unet = DDP(denoising_unet, device_ids=[local_rank], output_device=local_rank)

    # 计算每个epoch的更新步数和总训练轮数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    num_train_epochs = math.ceil(config.run.max_train_steps / num_update_steps_per_epoch)

    # 计算总batch size
    total_batch_size = config.data.batch_size * num_processes

    # 打印训练信息
    if is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {config.data.batch_size}")
        logger.info(f"  Total train batch size (w. parallel & distributed) = {total_batch_size}")
        logger.info(f"  Total optimization steps = {config.run.max_train_steps}")

    global_step = resume_global_step
    first_epoch = resume_global_step // num_update_steps_per_epoch

    # 创建进度条
    progress_bar = tqdm(
        range(0, config.run.max_train_steps),
        initial=resume_global_step,
        desc="Steps",
        disable=not is_main_process,
    )

    # 初始化训练记录列表
    train_step_list = []
    val_step_list = []
    sync_conf_list = []

    # 配置混合精度训练
    scaler = torch.amp.GradScaler("cuda") if config.run.mixed_precision_training else None

    # 开始训练循环
    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        denoising_unet.train()

        for step, batch in enumerate(train_dataloader):
            ### >>>> 训练阶段 >>>> ###

            # 处理音频输入
            if config.model.add_audio_layer:
                if batch["mel"] != []:
                    mel = batch["mel"].to(device, dtype=torch.float16)

                audio_embeds_list = []
                try:
                    for idx in range(len(batch["video_path"])):
                        video_path = batch["video_path"][idx]
                        start_idx = batch["start_idx"][idx]

                        with torch.no_grad():
                            audio_feat = audio_encoder.audio2feat(video_path)
                        audio_embeds = audio_encoder.crop_overlap_audio_window(audio_feat, start_idx)
                        audio_embeds_list.append(audio_embeds)
                except Exception as e:
                    logger.info(f"{type(e).__name__} - {e} - {video_path}")
                    continue
                audio_embeds = torch.stack(audio_embeds_list)  # (B, 16, 50, 384)
                audio_embeds = audio_embeds.to(device, dtype=torch.float16)
            else:
                audio_embeds = None

            # 将视频转换到潜空间
            gt_pixel_values = batch["gt_pixel_values"].to(device, dtype=torch.float16)
            masked_pixel_values = batch["masked_pixel_values"].to(device, dtype=torch.float16)
            masks = batch["masks"].to(device, dtype=torch.float16)
            ref_pixel_values = batch["ref_pixel_values"].to(device, dtype=torch.float16)

            # 重排张量维度
            gt_pixel_values = rearrange(gt_pixel_values, "b f c h w -> (b f) c h w")
            masked_pixel_values = rearrange(masked_pixel_values, "b f c h w -> (b f) c h w")
            masks = rearrange(masks, "b f c h w -> (b f) c h w")
            ref_pixel_values = rearrange(ref_pixel_values, "b f c h w -> (b f) c h w")

            # 使用VAE编码图像
            with torch.no_grad():
                gt_latents = vae.encode(gt_pixel_values).latent_dist.sample()
                masked_latents = vae.encode(masked_pixel_values).latent_dist.sample()
                ref_latents = vae.encode(ref_pixel_values).latent_dist.sample()

            # 调整mask大小
            masks = torch.nn.functional.interpolate(masks, size=config.data.resolution // vae_scale_factor)

            # 处理潜空间表示
            gt_latents = (
                rearrange(gt_latents, "(b f) c h w -> b c f h w", f=config.data.num_frames) - vae.config.shift_factor
            ) * vae.config.scaling_factor
            masked_latents = (
                rearrange(masked_latents, "(b f) c h w -> b c f h w", f=config.data.num_frames)
                - vae.config.shift_factor
            ) * vae.config.scaling_factor
            ref_latents = (
                rearrange(ref_latents, "(b f) c h w -> b c f h w", f=config.data.num_frames) - vae.config.shift_factor
            ) * vae.config.scaling_factor
            masks = rearrange(masks, "(b f) c h w -> b c f h w", f=config.data.num_frames)

            # 生成噪声
            if config.run.use_mixed_noise:
                # 参考论文: https://arxiv.org/abs/2305.10474
                noise_shared_std_dev = (config.run.mixed_noise_alpha**2 / (1 + config.run.mixed_noise_alpha**2)) ** 0.5
                noise_shared = torch.randn_like(gt_latents) * noise_shared_std_dev
                noise_shared = noise_shared[:, :, 0:1].repeat(1, 1, config.data.num_frames, 1, 1)

                noise_ind_std_dev = (1 / (1 + config.run.mixed_noise_alpha**2)) ** 0.5
                noise_ind = torch.randn_like(gt_latents) * noise_ind_std_dev
                noise = noise_ind + noise_shared
            else:
                noise = torch.randn_like(gt_latents)
                noise = noise[:, :, 0:1].repeat(
                    1, 1, config.data.num_frames, 1, 1
                )  # 对所有帧使用相同的噪声，参考论文: https://arxiv.org/abs/2308.09716

            bsz = gt_latents.shape[0]

            # 为每个视频采样随机时间步
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=gt_latents.device)
            timesteps = timesteps.long()

            # 根据时间步的噪声幅度向潜变量添加噪声
            # (这是前向扩散过程)
            noisy_gt_latents = noise_scheduler.add_noise(gt_latents, noise, timesteps)

            # 根据预测类型获取目标
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # 准备UNet输入
            denoising_unet_input = torch.cat([noisy_gt_latents, masks, masked_latents, ref_latents], dim=1)

            # 预测噪声并计算损失
            # 混合精度训练
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.run.mixed_precision_training):
                pred_noise = denoising_unet(denoising_unet_input, timesteps, encoder_hidden_states=audio_embeds).sample

            # 计算重建损失
            if config.run.recon_loss_weight != 0:
                recon_loss = F.mse_loss(pred_noise.float(), target.float(), reduction="mean")
            else:
                recon_loss = 0

            # 单步采样
            pred_latents = one_step_sampling(noise_scheduler, pred_noise, timesteps, noisy_gt_latents)

            # 如果在像素空间进行监督
            if config.run.pixel_space_supervise:
                pred_pixel_values = vae.decode(
                    rearrange(pred_latents, "b c f h w -> (b f) c h w") / vae.config.scaling_factor
                    + vae.config.shift_factor
                ).sample

            # 计算感知损失
            if config.run.perceptual_loss_weight != 0 and config.run.pixel_space_supervise:
                pred_pixel_values_perceptual = pred_pixel_values[:, :, pred_pixel_values.shape[2] // 2 :, :]
                gt_pixel_values_perceptual = gt_pixel_values[:, :, gt_pixel_values.shape[2] // 2 :, :]
                lpips_loss = lpips_loss_func(
                    pred_pixel_values_perceptual.float(), gt_pixel_values_perceptual.float()
                ).mean()
            else:
                lpips_loss = 0

            # 计算TREPA损失
            if config.run.trepa_loss_weight != 0 and config.run.pixel_space_supervise:
                trepa_pred_pixel_values = rearrange(
                    pred_pixel_values, "(b f) c h w -> b c f h w", f=config.data.num_frames
                )
                trepa_gt_pixel_values = rearrange(
                    gt_pixel_values, "(b f) c h w -> b c f h w", f=config.data.num_frames
                )
                trepa_loss = trepa_loss_func(trepa_pred_pixel_values, trepa_gt_pixel_values)
            else:
                trepa_loss = 0

            # 计算SyncNet损失
            if config.model.add_audio_layer and config.run.use_syncnet:
                if config.run.pixel_space_supervise:
                    syncnet_input = rearrange(
                        pred_pixel_values, "(b f) c h w -> b (f c) h w", f=config.data.num_frames
                    )
                else:
                    syncnet_input = rearrange(pred_latents, "b c f h w -> b (f c) h w")

                if syncnet_config.data.lower_half:
                    height = syncnet_input.shape[2]
                    syncnet_input = syncnet_input[:, :, height // 2 :, :]
                ones_tensor = torch.ones((config.data.batch_size, 1)).float().to(device=device)
                vision_embeds, audio_embeds = syncnet(syncnet_input, mel)
                sync_loss = cosine_loss(vision_embeds.float(), audio_embeds.float(), ones_tensor).mean()
            else:
                sync_loss = 0

            # 计算总损失
            loss = (
                recon_loss * config.run.recon_loss_weight
                + sync_loss * config.run.sync_loss_weight
                + lpips_loss * config.run.perceptual_loss_weight
                + trepa_loss * config.run.trepa_loss_weight
            )

            train_step_list.append(global_step)

            # 反向传播
            optimizer.zero_grad()

            if config.run.mixed_precision_training:
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, config.optimizer.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(trainable_params, config.optimizer.max_grad_norm)
                optimizer.step()

            # 更新学习率
            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            ### <<<< 训练阶段 <<<< ###

            # 保存检查点和进行验证
            if is_main_process and (global_step % config.ckpt.save_ckpt_steps == 0):
                model_save_path = os.path.join(output_dir, f"checkpoints/checkpoint-{global_step}.pt")
                state_dict = {
                    "global_step": global_step,
                    "state_dict": denoising_unet.module.state_dict(),
                }
                try:
                    torch.save(state_dict, model_save_path)
                    logger.info(f"Saved checkpoint to {model_save_path}")
                except Exception as e:
                    logger.error(f"Error saving model: {e}")

                # 验证
                logger.info("Running validation... ")

                validation_video_out_path = os.path.join(output_dir, f"val_videos/val_video_{global_step}.mp4")
                validation_video_mask_path = os.path.join(output_dir, f"val_videos/val_video_mask.mp4")

                # 使用混合精度进行验证
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pipeline(
                        config.data.val_video_path,
                        config.data.val_audio_path,
                        validation_video_out_path,
                        validation_video_mask_path,
                        num_frames=config.data.num_frames,
                        num_inference_steps=config.run.inference_steps,
                        guidance_scale=config.run.guidance_scale,
                        weight_dtype=torch.float16,
                        width=config.data.resolution,
                        height=config.data.resolution,
                        mask=config.data.mask,
                        mask_image_path=config.data.mask_image_path,
                    )

                # 评估同步置信度
                sync_conf = syncnet_eval(
                    validation_video_out_path,
                    config.data.val_audio_path,
                    syncnet_eval_model,
                    syncnet_detector,
                    os.path.join(output_dir, f"sync_conf_results/sync_conf_{global_step}.txt"),
                )
                sync_conf_list.append(sync_conf)

                # 绘制损失曲线
                plot_loss_chart(
                    train_step_list,
                    train_loss_list,
                    val_step_list,
                    val_loss_list,
                    os.path.join(output_dir, "loss_charts"),
                )

            if global_step >= config.run.max_train_steps:
                break

        if global_step >= config.run.max_train_steps:
            break

    # 训练结束，保存最终检查点
    if is_main_process:
        model_save_path = os.path.join(output_dir, "checkpoints/final_checkpoint.pt")
        state_dict = {
            "global_step": global_step,
            "state_dict": denoising_unet.module.state_dict(),
        }
        try:
            torch.save(state_dict, model_save_path)
            logger.info(f"Saved final checkpoint to {model_save_path}")
        except Exception as e:
            logger.error(f"Error saving final model: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Config file path
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")

    args = parser.parse_args()
    config = OmegaConf.load(args.unet_config_path)
    config.unet_config_path = args.unet_config_path

    main(config)
