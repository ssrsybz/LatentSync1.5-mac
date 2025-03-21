import zlib
from typing import Iterator, TextIO


def exact_div(x, y):
    """整除函数，确保x能被y整除
    
    Args:
        x: 被除数
        y: 除数
        
    Returns:
        整除的结果
        
    Raises:
        AssertionError: 当x不能被y整除时抛出异常
    """
    assert x % y == 0
    return x // y


def str2bool(string):
    """将字符串转换为布尔值
    
    Args:
        string: 要转换的字符串，必须是'True'或'False'
        
    Returns:
        对应的布尔值
        
    Raises:
        ValueError: 当输入字符串不是'True'或'False'时抛出异常
    """
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string):
    """将字符串转换为可选整数
    
    Args:
        string: 要转换的字符串
        
    Returns:
        如果字符串是'None'则返回None，否则返回转换后的整数
    """
    return None if string == "None" else int(string)


def optional_float(string):
    """将字符串转换为可选浮点数
    
    Args:
        string: 要转换的字符串
        
    Returns:
        如果字符串是'None'则返回None，否则返回转换后的浮点数
    """
    return None if string == "None" else float(string)


def compression_ratio(text) -> float:
    """计算文本的压缩比
    
    使用zlib压缩算法计算文本的压缩比，用于评估文本的信息密度
    
    Args:
        text: 要计算压缩比的文本
        
    Returns:
        压缩比，原文本长度除以压缩后的长度
    """
    return len(text) / len(zlib.compress(text.encode("utf-8")))


def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    """将秒数格式化为时间戳字符串
    
    Args:
        seconds: 要格式化的秒数
        always_include_hours: 是否始终包含小时部分，即使小时为0
        decimal_marker: 小数点标记符号
        
    Returns:
        格式化后的时间戳字符串，格式为HH:MM:SS.mmm或MM:SS.mmm
        
    Raises:
        AssertionError: 当seconds为负数时抛出异常
    """
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    # 计算小时数
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    # 计算分钟数
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    # 计算秒数
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    # 根据设置决定是否显示小时部分
    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"


def write_txt(transcript: Iterator[dict], file: TextIO):
    """将转录文本写入TXT文件
    
    Args:
        transcript: 包含转录文本段的迭代器
        file: 要写入的文件对象
    """
    for segment in transcript:
        print(segment['text'].strip(), file=file, flush=True)


def write_vtt(transcript: Iterator[dict], file: TextIO):
    """将转录文本写入VTT格式字幕文件
    
    Args:
        transcript: 包含转录文本段的迭代器，每个文本段包含start、end和text字段
        file: 要写入的文件对象
    """
    print("WEBVTT\n", file=file)
    for segment in transcript:
        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def write_srt(transcript: Iterator[dict], file: TextIO):
    """将转录文本写入SRT格式字幕文件
    
    Args:
        transcript: 包含转录文本段的迭代器，每个文本段包含start、end和text字段
        file: 要写入的文件对象
        
    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt

        result = transcribe(model, audio_path, temperature=temperature, **args)

        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    for i, segment in enumerate(transcript, start=1):
        # 写入SRT格式的字幕行
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )
