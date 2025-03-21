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

# 此模块用于移除人脸对齐失败的视频
# 主要功能包括：
# 1. 使用MediaPipe检测视频中的人脸
# 2. 验证每一帧是否只包含一个人脸
# 3. 删除不符合要求的视频
# 4. 支持多进程并行处理大量视频

import mediapipe as mp
from latentsync.utils.util import read_video, gather_video_paths_recursively
import os
import tqdm
from multiprocessing import Pool


class FaceDetector:
    """人脸检测类，用于验证视频中的人脸"""
    def __init__(self):
        """初始化人脸检测器"""
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

    def detect_face(self, image):
        """检测单帧图像中的人脸
        Args:
            image: 输入图像
        Returns:
            bool: 是否只包含一个人脸
        """
        # 处理图像并检测人脸
        results = self.face_detection.process(image)

        if not results.detections:  # 未检测到人脸
            return False

        if len(results.detections) != 1:  # 检测到多个人脸
            return False
        return True

    def detect_video(self, video_path):
        """检测视频中的所有帧
        Args:
            video_path: 输入视频路径
        Returns:
            bool: 所有帧是否都只包含一个人脸
        """
        try:
            video_frames = read_video(video_path, change_fps=False)
        except Exception as e:
            print(f"Exception: {e} - {video_path}")
            return False
        if len(video_frames) == 0:
            return False
        for frame in video_frames:
            if not self.detect_face(frame):
                return False
        return True

    def close(self):
        """关闭人脸检测器，释放资源"""
        self.face_detection.close()


def remove_incorrect_affined(video_path):
    """处理单个视频，如果不符合要求则删除
    Args:
        video_path: 输入视频路径
    """
    if not os.path.isfile(video_path):
        return
    face_detector = FaceDetector()
    has_face = face_detector.detect_video(video_path)
    if not has_face:
        os.remove(video_path)
        print(f"Removed: {video_path}")
    face_detector.close()


def remove_incorrect_affined_multiprocessing(input_dir, num_workers):
    """使用多进程并行处理视频
    Args:
        input_dir: 输入视频目录
        num_workers: 工作进程数
    """
    video_paths = gather_video_paths_recursively(input_dir)
    print(f"Total videos: {len(video_paths)}")

    print(f"Removing incorrect affined videos in {input_dir} ...")
    with Pool(num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(remove_incorrect_affined, video_paths), total=len(video_paths)):
            pass


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/multilingual_dcc/high_visual_quality"
    num_workers = 50  # 并行处理的进程数

    remove_incorrect_affined_multiprocessing(input_dir, num_workers)
