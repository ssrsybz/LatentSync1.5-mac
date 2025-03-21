import gradio as gr
from pathlib import Path
from scripts.inference import main
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import torch

# 模型配置文件和预训练权重路径
CONFIG_PATH = Path("configs/unet/stage2.yaml")  # UNet第二阶段模型配置文件
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")  # 预训练模型权重文件路径

# 检测可用的计算设备
def get_available_device():
    if torch.backends.mps.is_available():
        return "mps", "Apple Silicon GPU (MPS)"
    elif torch.cuda.is_available():
        return "cuda", f"NVIDIA GPU ({torch.cuda.get_device_name()})"
    else:
        return "cpu", "CPU (性能可能受限)"


def process_video(
    video_path,
    audio_path,
    guidance_scale,
    inference_steps,
    seed,
):
    """
    视频处理主函数
    :param video_path: 输入视频路径
    :param audio_path: 输入音频路径
    :param guidance_scale: 引导系数，控制生成效果与条件的一致性
    :param inference_steps: 推理步数，影响生成质量与速度
    :param seed: 随机种子，保证结果可复现
    :return: 处理后的视频文件路径
    """
    # Create the temp directory if it doesn't exist
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute Path objects and normalize them
    video_file_path = Path(video_path)
    video_path = video_file_path.absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Set the output path for the processed video
    output_path = str(output_dir / f"{video_file_path.stem}_{current_time}.mp4")  # Change the filename as needed

    config = OmegaConf.load(CONFIG_PATH)

    # 获取当前可用的计算设备
    device, _ = get_available_device()
    
    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
            "device": device,  # 添加设备配置
        }
    )

    # Parse the arguments
    args = create_args(video_path, audio_path, output_path, inference_steps, guidance_scale, seed)

    try:
        result = main(
            config=config,
            args=args,
        )
        print("Processing completed successfully.")
        return output_path  # Ensure the output path is returned
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise gr.Error(f"Error during processing: {str(e)}")


def create_args(
    video_path: str, audio_path: str, output_path: str, inference_steps: int, guidance_scale: float, seed: int
) -> argparse.Namespace:
    """
    构建命令行参数解析器
    返回包含模型运行所需参数的Namespace对象
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            CHECKPOINT_PATH.absolute().as_posix(),
            "--video_path",
            video_path,
            "--audio_path",
            audio_path,
            "--video_out_path",
            output_path,
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
        ]
    )


# Create Gradio interface
# 构建Gradio交互界面
with gr.Blocks(title="LatentSync Video Processing") as demo:
    """
    主界面布局包含：
    - 论文标题和作者信息
    - 输入输出视频/音频组件
    - 参数调节滑动条
    - 示例演示区域
    """
    # 获取当前可用的计算设备
    device, device_name = get_available_device()
    
    gr.Markdown(
        f"""
    # LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision
    Upload a video and audio file to process with LatentSync model.

    <div align="center">
        <strong>Chunyu Li1,2  Chao Zhang1  Weikai Xu1  Jinghui Xie1,†  Weiguo Feng1
        Bingyue Peng1  Weiwei Xing2,†</strong>
    </div>

    <div align="center">
        <strong>1ByteDance   2Beijing Jiaotong University</strong>
    </div>

    <div style="display:flex;justify-content:center;column-gap:4px;">
        <a href="https://github.com/bytedance/LatentSync">
            <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
        </a> 
        <a href="https://arxiv.org/pdf/2412.09262">
            <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
        </a>
    </div>

    <div align="center" style="margin-top:10px;padding:10px;background-color:#f0f0f0;border-radius:5px;">
        <strong>当前使用的计算设备：{device_name}</strong>
    </div>
    """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="输入视频")  # 视频上传组件
            audio_input = gr.Audio(label="输入音频", type="filepath")  # 音频上传组件

            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=2.5,
                    value=1.5,
                    step=0.5,
                    label="引导尺度",  # 控制生成效果的引导强度参数
                )
                inference_steps = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="Inference Steps")

            with gr.Row():
                seed = gr.Number(value=1247, label="Random Seed", precision=0)

            process_btn = gr.Button("Process Video")

        with gr.Column():
            video_output = gr.Video(label="Output Video")

            gr.Examples(
                examples=[
                    ["assets/demo1_video.mp4", "assets/demo1_audio.wav"],
                    ["assets/demo2_video.mp4", "assets/demo2_audio.wav"],
                    ["assets/demo3_video.mp4", "assets/demo3_audio.wav"],
                ],
                inputs=[video_input, audio_input],
            )

    process_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            audio_input,
            guidance_scale,
            inference_steps,
            seed,
        ],
        outputs=video_output,
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True)
