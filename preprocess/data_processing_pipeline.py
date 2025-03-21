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

"""
视频数据预处理流水线

该模块实现了一个完整的视频数据预处理流水线，包含以下主要步骤：
1. 清理损坏的视频文件
2. 统一视频帧率和音频采样率
3. 视频镜头检测和分割
4. 人脸对齐变换
5. 音视频同步检测
6. 视觉质量过滤

所有处理步骤都支持多进程或多GPU并行处理以提高效率。
输出的处理结果按步骤保存在不同的子目录中。
"""

import argparse
import os
from preprocess.affine_transform import affine_transform_multi_gpus
from preprocess.remove_broken_videos import remove_broken_videos_multiprocessing
from preprocess.detect_shot import detect_shot_multiprocessing
from preprocess.filter_high_resolution import filter_high_resolution_multiprocessing
from preprocess.resample_fps_hz import resample_fps_hz_multiprocessing
from preprocess.segment_videos import segment_videos_multiprocessing
from preprocess.sync_av import sync_av_multi_gpus
from preprocess.filter_visual_quality import filter_visual_quality_multi_gpus
from preprocess.remove_incorrect_affined import remove_incorrect_affined_multiprocessing


def data_processing_pipeline(
    total_num_workers, per_gpu_num_workers, resolution, sync_conf_threshold, temp_dir, input_dir
):
    """执行完整的视频数据预处理流水线

    Args:
        total_num_workers (int): CPU多进程的总进程数
        per_gpu_num_workers (int): 每个GPU上的工作进程数
        resolution (int): 视频分辨率，用于人脸对齐
        sync_conf_threshold (int): 音视频同步的置信度阈值
        temp_dir (str): 临时文件目录
        input_dir (str): 输入视频目录
    """
    # 第1步：移除损坏的视频文件
    print("Removing broken videos...")
    remove_broken_videos_multiprocessing(input_dir, total_num_workers)

    # 第2步：统一视频帧率(25fps)和音频采样率(16kHz)
    print("Resampling FPS hz...")
    resampled_dir = os.path.join(os.path.dirname(input_dir), "resampled")
    resample_fps_hz_multiprocessing(input_dir, resampled_dir, total_num_workers)

    # 第3步：检测视频中的镜头切换点
    print("Detecting shot...")
    shot_dir = os.path.join(os.path.dirname(input_dir), "shot")
    detect_shot_multiprocessing(resampled_dir, shot_dir, total_num_workers)

    # 第4步：根据镜头切换点分割视频
    print("Segmenting videos...")
    segmented_dir = os.path.join(os.path.dirname(input_dir), "segmented")
    segment_videos_multiprocessing(shot_dir, segmented_dir, total_num_workers)

    # 高分辨率过滤(可选步骤)
    # print("Filtering high resolution...")
    # high_resolution_dir = os.path.join(os.path.dirname(input_dir), "high_resolution")
    # filter_high_resolution_multiprocessing(segmented_dir, high_resolution_dir, resolution, total_num_workers)

    # 第5步：对视频进行人脸对齐变换，使用多GPU加速
    print("Affine transforming videos...")
    affine_transformed_dir = os.path.join(os.path.dirname(input_dir), "affine_transformed")
    affine_transform_multi_gpus(
        segmented_dir, affine_transformed_dir, temp_dir, resolution, per_gpu_num_workers // 2
    )

    # 第6步：移除人脸对齐失败的视频
    print("Removing incorrect affined videos...")
    remove_incorrect_affined_multiprocessing(affine_transformed_dir, total_num_workers)

    # 第7步：检测音视频是否同步，使用多GPU加速
    print("Syncing audio and video...")
    av_synced_dir = os.path.join(os.path.dirname(input_dir), f"av_synced_{sync_conf_threshold}")
    sync_av_multi_gpus(affine_transformed_dir, av_synced_dir, temp_dir, per_gpu_num_workers, sync_conf_threshold)

    # 第8步：过滤视觉质量较差的视频，使用多GPU加速
    print("Filtering visual quality...")
    high_visual_quality_dir = os.path.join(os.path.dirname(input_dir), "high_visual_quality")
    filter_visual_quality_multi_gpus(av_synced_dir, high_visual_quality_dir, per_gpu_num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 配置CPU多进程数，默认100个进程
    parser.add_argument("--total_num_workers", type=int, default=100)
    # 配置每个GPU的工作进程数，默认20个进程
    parser.add_argument("--per_gpu_num_workers", type=int, default=20)
    # 配置视频分辨率，默认256x256
    parser.add_argument("--resolution", type=int, default=256)
    # 配置音视频同步的置信度阈值，默认为3
    parser.add_argument("--sync_conf_threshold", type=int, default=3)
    # 配置临时文件目录
    parser.add_argument("--temp_dir", type=str, default="temp")
    # 配置输入视频目录(必需参数)
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    data_processing_pipeline(
        args.total_num_workers,
        args.per_gpu_num_workers,
        args.resolution,
        args.sync_conf_threshold,
        args.temp_dir,
        args.input_dir,
    )
