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
视频文件列表生成工具

该脚本用于递归收集指定目录下的视频文件路径，并将路径写入文件列表。
"""
from tqdm import tqdm
from latentsync.utils.util import gather_video_paths_recursively


class FileslistWriter:
    """
    文件列表写入器类
    
    功能:
        1. 初始化文件列表
        2. 递归收集视频文件路径
        3. 将路径追加到文件列表
    """
    def __init__(self, fileslist_path: str):
        """
        初始化文件列表写入器
        
        参数:
            fileslist_path: 文件列表保存路径
        """
        self.fileslist_path = fileslist_path
        with open(fileslist_path, "w") as _:
            pass

    def append_dataset(self, dataset_dir: str):
        """
        将数据集中的视频路径追加到文件列表
        
        参数:
            dataset_dir: 数据集目录路径
        """
        print(f"Dataset dir: {dataset_dir}")
        video_paths = gather_video_paths_recursively(dataset_dir)
        with open(self.fileslist_path, "a") as f:
            for video_path in tqdm(video_paths):
                f.write(f"{video_path}\n")


if __name__ == "__main__":
    """
    示例用法
    """
    fileslist_path = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/fileslist/data_v9_syncnet.txt"

    writer = FileslistWriter(fileslist_path)
    writer.append_dataset("/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/high_visual_quality/train")
    writer.append_dataset("/mnt/bn/maliva-gen-ai-v2/chunyu.li/HDTF/high_visual_quality/train")
