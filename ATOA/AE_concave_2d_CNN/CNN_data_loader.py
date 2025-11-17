import numpy as np
import os

def load_dataset_mask(N=120, split="train"):
    """
    从事先处理好的 .npy 文件中加载环境 mask。
    保持与旧函数相同的返回类型和形状 (N, H, W)。

    参数:
        N: 需要加载的环境数量
        NP: 保留以兼容旧接口（unused）
        split: 数据集名称 train/val/test
    """

    # 新的数据路径
    BASE_DIR = "data/random_2d/processed_all"
    MASK_FILE = os.path.join(BASE_DIR, split, "env_masks.npy")

    # 读取完整 env_masks
    if not os.path.exists(MASK_FILE):
        raise FileNotFoundError(f"找不到 mask 文件: {MASK_FILE}")

    env_masks_all = np.load(MASK_FILE, allow_pickle=False).astype(np.float32)

    # 原函数默认返回前 N 个
    N = min(N, len(env_masks_all))


    enviro_mask_set = env_masks_all[:N]

    return enviro_mask_set
