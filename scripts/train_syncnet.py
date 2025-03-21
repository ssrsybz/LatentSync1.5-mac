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

# 导入必要的库
from tqdm.auto import tqdm  # 进度条显示
import os, argparse, datetime, math
import logging
from omegaconf import OmegaConf  # 配置文件管理
import shutil

# 导入自定义模块
from latentsync.data.syncnet_dataset import SyncNetDataset  # 数据集类
from latentsync.models.stable_syncnet import StableSyncNet  # SyncNet模型
from latentsync.models.wav2lip_syncnet import Wav2LipSyncNet  # Wav2Lip版本的SyncNet模型
from latentsync.utils.util import gather_loss, plot_loss_chart  # 工具函数
from accelerate.utils import set_seed  # 随机种子设置

# 导入PyTorch相关库
import torch
from diffusers import AutoencoderKL  # VAE自编码器
from diffusers.utils.logging import get_logger
from einops import rearrange  # 张量维度重排
import torch.distributed as dist  # 分布式训练
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP封装
from torch.utils.data.distributed import DistributedSampler  # 分布式采样器
from latentsync.utils.util import init_dist, cosine_loss  # 分布式初始化和余弦损失

logger = get_logger(__name__)


def main(config):
    # 初始化分布式训练环境
    local_rank = init_dist()  # 获取本地rank
    global_rank = dist.get_rank()  # 获取全局rank
    num_processes = dist.get_world_size()  # 获取总进程数
    is_main_process = global_rank == 0  # 判断是否为主进程

    # 设置随机种子
    seed = config.run.seed + global_rank
    set_seed(seed)

    # 创建日志文件夹
    folder_name = "train" + datetime.datetime.now().strftime(f"-%Y_%m_%d-%H:%M:%S")
    output_dir = os.path.join(config.data.train_output_dir, folder_name)

    # 配置日志格式
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # 创建输出目录（仅在主进程中执行）
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)  # 检查点保存目录
        os.makedirs(f"{output_dir}/loss_charts", exist_ok=True)  # 损失曲线保存目录
        shutil.copy(config.config_path, output_dir)  # 复制配置文件

    device = torch.device(local_rank)

    # 如果在潜空间中训练，加载VAE模型
    if config.data.latent_space:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
        vae.requires_grad_(False)  # 冻结VAE参数
        vae.to(device)
    else:
        vae = None

    # 创建训练和验证数据集
    train_dataset = SyncNetDataset(config.data.train_data_dir, config.data.train_fileslist, config)
    val_dataset = SyncNetDataset(config.data.val_data_dir, config.data.val_fileslist, config)

    # 创建分布式采样器
    train_distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=config.run.seed,
    )

    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,  # 使用DistributedSampler时shuffle必须为False
        sampler=train_distributed_sampler,
        num_workers=config.data.num_workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn,
    )

    # 限制验证集batch size以避免显存溢出
    num_samples_limit = 640
    val_batch_size = min(
        num_samples_limit // config.data.num_frames, config.data.batch_size
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=val_dataset.worker_init_fn,
    )

    # 初始化模型
    syncnet = StableSyncNet(OmegaConf.to_container(config.model)).to(device)
    # syncnet = Wav2LipSyncNet().to(device)  # 可选的Wav2Lip版本

    # 配置优化器
    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, syncnet.parameters())), lr=config.optimizer.lr
    )

    # 加载检查点（如果存在）
    if config.ckpt.resume_ckpt_path != "":
        if is_main_process:
            logger.info(f"Load checkpoint from: {config.ckpt.resume_ckpt_path}")
        ckpt = torch.load(config.ckpt.resume_ckpt_path, map_location=device, weights_only=True)

        syncnet.load_state_dict(ckpt["state_dict"])
        global_step = ckpt["global_step"]
        train_step_list = ckpt["train_step_list"]
        train_loss_list = ckpt["train_loss_list"]
        val_step_list = ckpt["val_step_list"]
        val_loss_list = ckpt["val_loss_list"]
    else:
        # 初始化训练状态
        global_step = 0
        train_step_list = []
        train_loss_list = []
        val_step_list = []
        val_loss_list = []

    # 使用DDP包装模型
    syncnet = DDP(syncnet, device_ids=[local_rank], output_device=local_rank)

    # 计算训练步数和轮数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    num_train_epochs = math.ceil(config.run.max_train_steps / num_update_steps_per_epoch)

    # 打印训练信息（仅在主进程中执行）
    if is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {config.data.batch_size}")
        logger.info(f"  Total train batch size (w. parallel & distributed) = {config.data.batch_size * num_processes}")
        logger.info(f"  Total optimization steps = {config.run.max_train_steps}")

    first_epoch = global_step // num_update_steps_per_epoch
    num_val_batches = config.data.num_val_samples // (num_processes * config.data.batch_size)

    # 创建进度条（仅在主进程中显示）
    progress_bar = tqdm(
        range(0, config.run.max_train_steps), initial=global_step, desc="Steps", disable=not is_main_process
    )

    # 配置混合精度训练
    scaler = torch.amp.GradScaler("cuda") if config.run.mixed_precision_training else None

    # 开始训练循环
    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)  # 设置采样器epoch
        syncnet.train()  # 设置为训练模式

        for step, batch in enumerate(train_dataloader):
            ### >>>> 训练阶段 >>>> ###

            # 将数据移到GPU并设置数据类型
            frames = batch["frames"].to(device, dtype=torch.float16)
            audio_samples = batch["audio_samples"].to(device, dtype=torch.float16)
            y = batch["y"].to(device, dtype=torch.float32)

            # 如果在潜空间中训练，使用VAE编码图像
            if config.data.latent_space:
                max_batch_size = num_samples_limit // config.data.num_frames
                # 如果batch size过大，分批处理以避免显存溢出
                if frames.shape[0] > max_batch_size:
                    assert (
                        frames.shape[0] % max_batch_size == 0
                    ), f"max_batch_size {max_batch_size} should be divisible by batch_size {frames.shape[0]}"
                    frames_part_results = []
                    for i in range(0, frames.shape[0], max_batch_size):
                        frames_part = frames[i : i + max_batch_size]
                        frames_part = rearrange(frames_part, "b f c h w -> (b f) c h w")
                        with torch.no_grad():
                            frames_part = vae.encode(frames_part).latent_dist.sample() * 0.18215
                        frames_part_results.append(frames_part)
                    frames = torch.cat(frames_part_results, dim=0)
                else:
                    frames = rearrange(frames, "b f c h w -> (b f) c h w")
                    with torch.no_grad():
                        frames = vae.encode(frames).latent_dist.sample() * 0.18215

                frames = rearrange(frames, "(b f) c h w -> b (f c) h w", f=config.data.num_frames)
            else:
                frames = rearrange(frames, "b f c h w -> b (f c) h w")

            # 如果只使用下半部分图像
            if config.data.lower_half:
                height = frames.shape[2]
                frames = frames[:, :, height // 2 :, :]

            # 使用混合精度训练
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.run.mixed_precision_training):
                vision_embeds, audio_embeds = syncnet(frames, audio_samples)

            # 计算损失
            loss = cosine_loss(vision_embeds.float(), audio_embeds.float(), y).mean()

            # 清空梯度
            optimizer.zero_grad()

            # 反向传播
            if config.run.mixed_precision_training:
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(syncnet.parameters(), config.optimizer.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(syncnet.parameters(), config.optimizer.max_grad_norm)
                optimizer.step()

            # 更新进度
            progress_bar.update(1)
            global_step += 1

            # 收集所有进程的损失并计算平均值
            global_average_loss = gather_loss(loss, device)
            train_step_list.append(global_step)
            train_loss_list.append(global_average_loss)

            # 定期验证（仅在主进程中执行）
            if is_main_process and global_step % config.run.validation_steps == 0:
                logger.info(f"Validation at step {global_step}")
                val_loss = validation(
                    val_dataloader,
                    device,
                    syncnet,
                    cosine_loss,
                    config.data.latent_space,
                    config.data.lower_half,
                    vae,
                    num_val_batches,
                )
                val_step_list.append(global_step)
                val_loss_list.append(val_loss)
                logger.info(f"Validation loss at step {global_step} is {val_loss:0.3f}")

            # 定期保存检查点（仅在主进程中执行）
            if is_main_process and global_step % config.ckpt.save_ckpt_steps == 0:
                checkpoint_save_path = os.path.join(output_dir, f"checkpoints/checkpoint-{global_step}.pt")
                torch.save(
                    {
                        "state_dict": syncnet.module.state_dict(),  # 解包DDP
                        "global_step": global_step,
                        "train_step_list": train_step_list,
                        "train_loss_list": train_loss_list,
                        "val_step_list": val_step_list,
                        "val_loss_list": val_loss_list,
                    },
                    checkpoint_save_path,
                )
                logger.info(f"Saved checkpoint to {checkpoint_save_path}")
                # 绘制损失曲线
                plot_loss_chart(
                    os.path.join(output_dir, f"loss_charts/loss_chart-{global_step}.png"),
                    ("Train loss", train_step_list, train_loss_list),
                    ("Val loss", val_step_list, val_loss_list),
                )

            # 更新进度条显示
            progress_bar.set_postfix({"step_loss": global_average_loss, "epoch": epoch})
            if global_step >= config.run.max_train_steps:
                break

    # 训练结束，清理资源
    progress_bar.close()
    dist.destroy_process_group()


@torch.no_grad()
def validation(val_dataloader, device, syncnet, cosine_loss, latent_space, lower_half, vae, num_val_batches):
    """验证函数
    
    Args:
        val_dataloader: 验证数据加载器
        device: 设备
        syncnet: 模型
        cosine_loss: 损失函数
        latent_space: 是否在潜空间中训练
        lower_half: 是否只使用下半部分图像
        vae: VAE模型
        num_val_batches: 验证批次数
    
    Returns:
        float: 平均验证损失
    """
    syncnet.eval()  # 设置为评估模式

    losses = []
    val_step = 0
    while True:
        for step, batch in enumerate(val_dataloader):
            ### >>>> 验证阶段 >>>> ###

            # 将数据移到GPU并设置数据类型
            frames = batch["frames"].to(device, dtype=torch.float16)
            audio_samples = batch["audio_samples"].to(device, dtype=torch.float16)
            y = batch["y"].to(device, dtype=torch.float32)

            # 如果在潜空间中验证，使用VAE编码图像
            if latent_space:
                num_frames = frames.shape[1]
                frames = rearrange(frames, "b f c h w -> (b f) c h w")
                frames = vae.encode(frames).latent_dist.sample() * 0.18215
                frames = rearrange(frames, "(b f) c h w -> b (f c) h w", f=num_frames)
            else:
                frames = rearrange(frames, "b f c h w -> b (f c) h w")

            # 如果只使用下半部分图像
            if lower_half:
                height = frames.shape[2]
                frames = frames[:, :, height // 2 :, :]

            # 使用混合精度
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                vision_embeds, audio_embeds = syncnet(frames, audio_samples)

            # 计算损失
            loss = cosine_loss(vision_embeds.float(), audio_embeds.float(), y).mean()

            losses.append(loss.item())

            val_step += 1
            if val_step > num_val_batches:
                syncnet.train()  # 恢复训练模式
                if len(losses) == 0:
                    raise RuntimeError("No validation data")
                return sum(losses) / len(losses)  # 返回平均损失


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Code to train the SyncNet")
    parser.add_argument("--config_path", type=str, default="configs/syncnet/syncnet_16_pixel.yaml")
    args = parser.parse_args()

    # 加载配置文件
    config = OmegaConf.load(args.config_path)
    config.config_path = args.config_path

    main(config)
