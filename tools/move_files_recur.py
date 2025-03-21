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
递归移动视频文件工具

该脚本用于递归地将输入目录中的视频文件移动到输出目录，保持目录结构不变。
"""
import os
import shutil
from tqdm import tqdm

# 存储待移动的文件路径列表
paths = []

def gather_paths(input_dir, output_dir):
    """
    递归收集视频文件路径
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的文件和子目录
    for video in sorted(os.listdir(input_dir)):
        # 处理视频文件
        if video.endswith(".mp4"):
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, video)
            # 跳过已存在的文件
            if os.path.isfile(video_output):
                continue
            # 记录待移动的文件路径
            paths.append([video_input, output_dir])
        # 递归处理子目录
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_paths(os.path.join(input_dir, video), os.path.join(output_dir, video))


def main(input_dir, output_dir):
    """
    主函数：递归移动视频文件
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    print(f"Recursively gathering video paths of {input_dir} ...")
    # 收集所有视频文件路径
    gather_paths(input_dir, output_dir)

    # 使用进度条移动文件
    for video_input, output_dir in tqdm(paths):
        shutil.move(video_input, output_dir)


if __name__ == "__main__":
    """
    示例用法
    """
    # 从input_dir移动到output_dir
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/willdata2"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/willdata"

    main(input_dir, output_dir)
