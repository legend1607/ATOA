# prepare_all_datasets.py
import os
import json
import numpy as np
from tqdm import tqdm
from skimage.draw import polygon

# ---------------- 参数配置 ----------------
BASE_DIR = "data/random_2d"  # 数据集总目录
SAVE_DIR = os.path.join(BASE_DIR, "processed_all")
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = (160, 160)  # CNN输入掩码大小
NUM_SECTORS = 8         # 方向分类数

DATASETS = ["train", "val", "test"]

# ---------------- 工具函数 ----------------
def create_env_mask(env_dict, img_size=(160,160)):
    H, W = img_size
    mask = np.zeros((H, W), dtype=np.float32)
    for rect in env_dict.get("rectangle_obstacles", []):
        x, y, w, h = rect
        xs = np.array([x, x + w, x + w, x])
        ys = np.array([y, y, y + h, y + h])
        xs = np.round(xs / env_dict["env_dims"][0] * (W-1)).astype(np.int32)
        ys = np.round(ys / env_dict["env_dims"][1] * (H-1)).astype(np.int32)
        xs = np.clip(xs, 0, W-1)
        ys = np.clip(ys, 0, H-1)
        rr, cc = polygon(ys, xs, shape=mask.shape)
        mask[rr, cc] = 1.0
    return mask

def compute_boundary_points(env_dict, num_points=50):
    points = []
    for rect in env_dict.get("rectangle_obstacles", []):
        x, y, w, h = rect
        for xi in np.linspace(x, x + w, num=num_points//4, endpoint=False):
            points.append([xi, y])
            points.append([xi, y + h])
        for yi in np.linspace(y, y + h, num=num_points//4, endpoint=False):
            points.append([x, yi])
            points.append([x + w, yi])
    return np.array(points, dtype=np.float32)

def direction_to_class(start, goal, num_sectors=NUM_SECTORS):
    vec = np.array(goal) - np.array(start)
    angle = np.arctan2(vec[1], vec[0])
    sector = int(((angle + np.pi) / (2 * np.pi)) * num_sectors)
    return sector % num_sectors

# ---------------- 主函数 ----------------
for dataset in DATASETS:
    DATA_DIR = os.path.join(BASE_DIR, dataset)
    JSON_FILE = os.path.join(DATA_DIR, "envs.json")
    PROCESSED_DIR = os.path.join(SAVE_DIR, dataset)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"\n🔹 处理数据集 [{dataset}]...")

    env_masks = []
    boundary_points_set = []
    boundary_lengths = []
    env_indices = []

    orient_dataset = []
    targets = []
    padded_targets_future_all = []
    classification_orient_targets = []
    classification_norm_targets = []

    max_future_length = 0

    with open(JSON_FILE, "r") as f:
        env_list = json.load(f)

    for env_idx, env in enumerate(tqdm(env_list)):
        mask = create_env_mask(env)
        env_masks.append(mask)

        boundary_points = compute_boundary_points(env)
        boundary_points_set.append(boundary_points)
        boundary_lengths.append(len(boundary_points))

        paths = env.get("paths", [])
        starts = env.get("start", [])
        goals = env.get("goal", [])

        for path, start, goal in zip(paths, starts, goals):
            path = np.array(path, dtype=np.float32)
            orient_dataset.append(np.concatenate([np.array(start), np.array(goal)]))
            targets.append(np.array(goal))
            padded_targets_future_all.append(path)
            max_future_length = max(max_future_length, path.shape[0])
            env_indices.append(env_idx)

            classification_orient_targets.append(direction_to_class(start, goal))
            classification_norm_targets.append(np.linalg.norm(np.array(goal) - np.array(start)))

    # pad future paths
    padded_array = np.zeros((len(padded_targets_future_all), max_future_length, 2), dtype=np.float32)
    for i, path in enumerate(padded_targets_future_all):
        padded_array[i, :path.shape[0], :] = path
    padded_targets_future_all = padded_array

    # 保存 npy 文件
    np.save(os.path.join(PROCESSED_DIR, "env_masks.npy"), np.array(env_masks, dtype=np.float32))
    np.save(os.path.join(PROCESSED_DIR, "boundary_points_set.npy"), np.array(boundary_points_set, dtype=object))
    np.save(os.path.join(PROCESSED_DIR, "boundary_lengths.npy"), np.array(boundary_lengths, dtype=np.int32))
    np.save(os.path.join(PROCESSED_DIR, "env_indices.npy"), np.array(env_indices, dtype=np.int32))
    np.save(os.path.join(PROCESSED_DIR, "orient_dataset.npy"), np.array(orient_dataset, dtype=np.float32))
    np.save(os.path.join(PROCESSED_DIR, "targets.npy"), np.array(targets, dtype=np.float32))
    np.save(os.path.join(PROCESSED_DIR, "padded_targets_future_all.npy"), padded_targets_future_all)
    np.save(os.path.join(PROCESSED_DIR, "classification_orient_targets.npy"), np.array(classification_orient_targets, dtype=np.int64))
    np.save(os.path.join(PROCESSED_DIR, "classification_norm_targets.npy"), np.array(classification_norm_targets, dtype=np.float32))

    print(f"✅ [{dataset}] 数据集处理完成，保存至 {PROCESSED_DIR}")
