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

# 此模块用于对视频进行人脸对齐变换处理
# 主要功能包括：
# 1. 检测视频中的人脸并进行仿射变换，使人脸对齐标准化
# 2. 支持多GPU并行处理大量视频
# 3. 保持音频与视频的同步

from latentsync.utils.util import read_video, write_video
from latentsync.utils.image_processor import ImageProcessor
import torch
from einops import rearrange
import os
import tqdm
import subprocess
from multiprocessing import Process
import shutil

# 存储所有待处理视频的输入和输出路径
paths = []


def gather_video_paths(input_dir, output_dir):
    """递归收集所有需要处理的视频路径
    Args:
        input_dir: 输入视频目录
        output_dir: 输出视频目录
    """
    for video in sorted(os.listdir(input_dir)):
        if video.endswith(".mp4"):
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, video)
            if os.path.isfile(video_output):
                continue
            paths.append((video_input, video_output))
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_video_paths(os.path.join(input_dir, video), os.path.join(output_dir, video))


class FaceDetector:
    """人脸检测和对齐处理类"""
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        """初始化人脸检测器
        Args:
            resolution: 输出视频的分辨率
            device: 运行设备，可以是'cpu'或'cuda:x'
        """
        self.image_processor = ImageProcessor(resolution, "fix_mask", device)

    def affine_transform_video(self, video_path):
        """对视频进行人脸对齐变换
        Args:
            video_path: 输入视频路径
        Returns:
            处理后的视频帧数组
        """
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            # 对每一帧进行人脸对齐变换，不允许多张人脸
            frame, _, _ = self.image_processor.affine_transform(frame, allow_multi_faces=False)
            results.append(frame)
        results = torch.stack(results)
        # 调整维度顺序：[帧数, 通道, 高, 宽] -> [帧数, 高, 宽, 通道]
        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results

    def close(self):
        """关闭人脸检测器，释放资源"""
        self.image_processor.close()


def combine_video_audio(video_frames, video_input_path, video_output_path, process_temp_dir):
    """合并处理后的视频帧和原始音频
    Args:
        video_frames: 处理后的视频帧数组
        video_input_path: 原始视频路径
        video_output_path: 输出视频路径
        process_temp_dir: 临时文件目录
    """
    video_name = os.path.basename(video_input_path)[:-4]
    audio_temp = os.path.join(process_temp_dir, f"{video_name}_temp.wav")
    video_temp = os.path.join(process_temp_dir, f"{video_name}_temp.mp4")

    # 保存处理后的视频帧
    write_video(video_temp, video_frames, fps=25)

    # 提取原始视频的音频
    command = f"ffmpeg -y -loglevel error -i {video_input_path} -q:a 0 -map a {audio_temp}"
    subprocess.run(command, shell=True)

    # 合并处理后的视频和原始音频
    os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
    command = f"ffmpeg -y -loglevel error -i {video_temp} -i {audio_temp} -c:v libx264 -c:a aac -map 0:v -map 1:a -q:v 0 -q:a 0 {video_output_path}"
    subprocess.run(command, shell=True)

    # 清理临时文件
    os.remove(audio_temp)
    os.remove(video_temp)


def func(paths, process_temp_dir, device_id, resolution):
    """单个进程的处理函数
    Args:
        paths: 待处理的视频路径列表
        process_temp_dir: 临时文件目录
        device_id: GPU设备ID
        resolution: 输出视频的分辨率
    """
    os.makedirs(process_temp_dir, exist_ok=True)
    face_detector = FaceDetector(resolution, f"cuda:{device_id}")

    for video_input, video_output in paths:
        if os.path.isfile(video_output):
            continue
        try:
            video_frames = face_detector.affine_transform_video(video_input)
        except Exception as e:  # 处理人脸检测失败的情况
            print(f"Exception: {e} - {video_input}")
            continue

        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        combine_video_audio(video_frames, video_input, video_output, process_temp_dir)
        print(f"Saved: {video_output}")

    face_detector.close()


def split(a, n):
    """将列表平均分割成n份
    Args:
        a: 输入列表
        n: 分割份数
    Returns:
        生成器，产生n个子列表
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def affine_transform_multi_gpus(input_dir, output_dir, temp_dir, resolution, num_workers):
    """使用多GPU并行处理视频的人脸对齐变换
    Args:
        input_dir: 输入视频目录
        output_dir: 输出视频目录
        temp_dir: 临时文件目录
        resolution: 输出视频的分辨率
        num_workers: 每个GPU上的工作进程数
    """
    print(f"Recursively gathering video paths of {input_dir} ...")
    gather_video_paths(input_dir, output_dir)
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No GPUs found")

    # 清理并创建临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # 将视频路径列表平均分配给所有工作进程
    split_paths = list(split(paths, num_workers * num_devices))

    processes = []

    # 为每个GPU创建多个工作进程
    for i in range(num_devices):
        for j in range(num_workers):
            process_index = i * num_workers + j
            process = Process(
                target=func, args=(split_paths[process_index], os.path.join(temp_dir, f"process_{i}"), i, resolution)
            )
            process.start()
            processes.append(process)

    # 等待所有进程完成
    for process in processes:
        process.join()


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/willdata2/segmented"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/willdata2/affine_transformed"
    temp_dir = "temp"
    resolution = 256  # 输出视频的分辨率
    num_workers = 10  # 每个GPU的工作进程数

    affine_transform_multi_gpus(input_dir, output_dir, temp_dir, resolution, num_workers)
