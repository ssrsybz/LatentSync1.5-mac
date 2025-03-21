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

'''
GPU显存占用工具
功能：通过持续进行矩阵运算保持GPU高占用率
应用场景：
- 防止其他进程占用GPU资源
- 测试GPU稳定性
- 保持显存分配防止碎片化
'''

import torch
import os
import torch.multiprocessing as mp
import time


def check_mem(cuda_device):
    """
    检测指定GPU的显存使用情况
    参数：
        cuda_device: GPU设备编号
    返回：
        total: 总显存（MB）
        used: 已用显存（MB）
    """
    # 执行nvidia-smi命令获取显存信息
    devices_info = (
        os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader')
        .read()
        .strip()
        .split("\n")
    )
    # 解析指定设备的显存数据
    total, used = devices_info[int(cuda_device)].split(",")
    return total, used


def loop(cuda_device):
    """
    持续占用GPU的循环函数
    参数：
        cuda_device: GPU设备编号
    """
    # 指定使用的CUDA设备
    cuda_i = torch.device(f"cuda:{cuda_device}")
    
    # 获取当前显存状态
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    
    # 计算目标占用显存（保留10%的显存空间）
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    
    # 持续进行矩阵运算
    while True:
        # 生成随机矩阵（20x512x512约占用20MB显存）
        x = torch.rand(20, 512, 512, dtype=torch.float, device=cuda_i)
        y = torch.rand(20, 512, 512, dtype=torch.float, device=cuda_i)
        
        # 短暂休眠避免过度消耗CPU
        time.sleep(0.001)
        
        # 执行矩阵乘法运算（显存密集型操作）
        x = torch.matmul(x, y)


def main():
    """主函数：启动多进程占用所有可用GPU"""
    if torch.cuda.is_available():
        # 获取可用GPU数量
        num_processes = torch.cuda.device_count()
        processes = list()
        
        # 为每个GPU创建独立进程
        for i in range(num_processes):
            p = mp.Process(target=loop, args=(i,))
            p.start()
            processes.append(p)
        
        # 等待所有进程结束（理论上会无限运行）
        for p in processes:
            p.join()


if __name__ == "__main__":
    # 设置多进程启动方式为spawn
    torch.multiprocessing.set_start_method("spawn")
    main()
