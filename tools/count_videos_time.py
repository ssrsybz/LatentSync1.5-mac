"""
视频时长统计工具

该脚本用于统计指定目录下所有视频的时长，并生成时长分布直方图。
"""
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

import matplotlib.pyplot as plt
from latentsync.utils.util import count_video_time, gather_video_paths_recursively
from tqdm import tqdm


def plot_histogram(data, fig_path):
    """
    绘制视频时长分布直方图
    
    参数:
        data: 视频时长数据列表
        fig_path: 保存直方图的文件路径
    """
    # 创建直方图
    plt.hist(data, bins=30, edgecolor="black")

    # 添加标题和标签
    plt.title("视频时长分布直方图")
    plt.xlabel("视频时长")
    plt.ylabel("频数")

    # 保存图像文件
    plt.savefig(fig_path)  # 保存为PNG文件，也可使用'histogram.jpg'、'histogram.pdf'等格式


def main(input_dir, fig_path):
    """
    主函数：统计视频时长并生成直方图
    
    参数:
        input_dir: 包含视频文件的输入目录
        fig_path: 保存直方图的文件路径
    """
    # 递归获取所有视频文件路径
    video_paths = gather_video_paths_recursively(input_dir)
    
    # 初始化视频时长列表
    video_times = []
    
    # 使用进度条遍历每个视频文件
    for video_path in tqdm(video_paths):
        # 计算并记录每个视频的时长
        video_times.append(count_video_time(video_path))
    
    # 绘制时长分布直方图
    plot_histogram(video_times, fig_path)


if __name__ == "__main__":
    """
    示例用法
    """
    input_dir = "validation"  # 输入目录
    fig_path = "histogram.png"  # 输出图像路径

    main(input_dir, fig_path)
