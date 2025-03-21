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

# 此模块用于筛选包含高分辨率人脸的视频
# 主要功能包括：
# 1. 使用MediaPipe检测视频中的人脸
# 2. 筛选出人脸区域分辨率达到要求的视频
# 3. 支持多进程并行处理大量视频

import mediapipe as mp
from latentsync.utils.util import read_video
import os
import tqdm
import shutil
from multiprocessing import Pool

# 存储所有待处理视频的路径信息
paths = []


def gather_video_paths(input_dir, output_dir, resolution):
    """递归收集所有需要处理的视频路径
    Args:
        input_dir: 输入视频目录
        output_dir: 输出视频目录
        resolution: 人脸分辨率要求
    """
    for video in sorted(os.listdir(input_dir)):
        if video.endswith(".mp4"):
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, video)
            if os.path.isfile(video_output):
                continue
            paths.append([video_input, video_output, resolution])
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_video_paths(os.path.join(input_dir, video), os.path.join(output_dir, video), resolution)


class FaceDetector:
    """人脸检测和分辨率验证类"""
    def __init__(self, resolution=256):
        """初始化人脸检测器
        Args:
            resolution: 人脸区域最小分辨率要求
        """
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.resolution = resolution

    def detect_face(self, image):
        """检测单帧图像中的人脸并验证分辨率
        Args:
            image: 输入图像
        Returns:
            bool: 是否包含符合要求的人脸
        Raises:
            Exception: 未检测到人脸时抛出异常
        """
        height, width = image.shape[:2]
        # 处理图像并检测人脸
        results = self.face_detection.process(image)

        if not results.detections:  # 未检测到人脸
            raise Exception("Face not detected")

        if len(results.detections) != 1:  # 检测到多个人脸
            return False
        detection = results.detections[0]  # 只使用第一个检测到的人脸

        # 计算人脸区域的实际分辨率
        bounding_box = detection.location_data.relative_bounding_box
        face_width = int(bounding_box.width * width)
        face_height = int(bounding_box.height * height)
        # 验证人脸区域是否达到最小分辨率要求
        if face_width < self.resolution or face_height < self.resolution:
            return False
        return True

    def detect_video(self, video_path):
        """检测视频中的所有帧
        Args:
            video_path: 输入视频路径
        Returns:
            bool: 所有帧是否都包含符合要求的人脸
        """
        video_frames = read_video(video_path, change_fps=False)
        if len(video_frames) == 0:
            return False
        for frame in video_frames:
            if not self.detect_face(frame):
                return False
        return True

    def close(self):
        """关闭人脸检测器，释放资源"""
        self.face_detection.close()


def filter_video(video_input, video_out, resolution):
    """处理单个视频
    Args:
        video_input: 输入视频路径
        video_out: 输出视频路径
        resolution: 人脸分辨率要求
    """
    if os.path.isfile(video_out):
        return
    face_detector = FaceDetector(resolution)
    try:
        save = face_detector.detect_video(video_input)
    except Exception as e:
        # print(f"Exception: {e} Input video: {video_input}")
        face_detector.close()
        return
    if save:
        os.makedirs(os.path.dirname(video_out), exist_ok=True)
        shutil.copy(video_input, video_out)
    face_detector.close()


def multi_run_wrapper(args):
    """多进程处理的包装函数
    Args:
        args: 包含视频处理参数的列表
    Returns:
        filter_video函数的返回值
    """
    return filter_video(*args)


def filter_high_resolution_multiprocessing(input_dir, output_dir, resolution, num_workers):
    """使用多进程并行处理视频
    Args:
        input_dir: 输入视频目录
        output_dir: 输出视频目录
        resolution: 人脸分辨率要求
        num_workers: 工作进程数
    """
    print(f"Recursively gathering video paths of {input_dir} ...")
    gather_video_paths(input_dir, output_dir, resolution)

    print(f"Filtering high resolution videos in {input_dir} ...")
    with Pool(num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(multi_run_wrapper, paths), total=len(paths)):
            pass


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai/lichunyu/HDTF/original/train"
    output_dir = "/mnt/bn/maliva-gen-ai/lichunyu/HDTF/detected/train"
    resolution = 256  # 人脸区域最小分辨率要求
    num_workers = 50  # 并行处理的进程数

    filter_high_resolution_multiprocessing(input_dir, output_dir, resolution, num_workers)
