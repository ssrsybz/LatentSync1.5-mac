"""
计算视频文件列表中所有视频的总时长

该脚本读取包含视频文件路径的列表文件，计算所有视频的总时长，并以小时为单位输出结果。
"""
from latentsync.utils.util import count_video_time
from tqdm import tqdm


def count_total_videos_time(fileslist_path: str):
    """
    计算视频文件列表中所有视频的总时长
    
    参数:
        fileslist_path: 包含视频文件路径的文本文件路径
    """
    # 读取文件列表
    with open(fileslist_path, "r") as f:
        filepaths = f.readlines()

    # 移除每行末尾的换行符
    filepaths = [filepath.strip() for filepath in filepaths]

    # 初始化总时长
    total_videos_time = 0
    
    # 使用进度条遍历每个视频文件
    for filepath in tqdm(filepaths):
        # 累加每个视频的时长
        total_videos_time += count_video_time(filepath)

    # 输出结果
    print(f"Fileslist path: {fileslist_path}")
    print(f"Total videos time: {round(total_videos_time/3600)} hours")


if __name__ == "__main__":
    # 示例用法
    fileslist_path = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/fileslist/data_v9_full.txt"
    count_total_videos_time(fileslist_path)
