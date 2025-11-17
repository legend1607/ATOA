import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import CNN_2d as AE
from skimage.draw import polygon

# ---------------- 配置 ----------------
JSON_FILE = "data/random_2d/test/envs.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (160, 160)
MODEL_PATH = "./models/cae"

def create_env_mask(env_dict, img_size=IMG_SIZE):
    H, W = img_size
    mask = np.zeros((H, W), dtype=np.float32)
    for rect in env_dict.get("rectangle_obstacles", []):
        x, y, w, h = rect
        xs = np.array([x, x + w, x + w, x])
        ys = np.array([y, y, y + h, y + h])
        xs = (xs / env_dict["env_dims"][0] * (W-1)).astype(np.int32)
        ys = (ys / env_dict["env_dims"][1] * (H-1)).astype(np.int32)
        rr, cc = polygon(ys, xs, shape=mask.shape)
        mask[rr, cc] = 1.0
    return mask

def draw_env(ax, env_dict):
    ax.set_xlim(0, env_dict["env_dims"][0])
    ax.set_ylim(0, env_dict["env_dims"][1])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    for rect in env_dict.get("rectangle_obstacles", []):
        x, y, w, h = rect
        ax.add_patch(plt.Rectangle((x,y), w,h, color='black'))
    ax.axis('off')

def draw_paths(ax, env_dict, img_size=IMG_SIZE, color='red'):
    """绘制路径到ax上，自动缩放"""
    H, W = img_size
    for path in env_dict.get("paths", []):
        path = np.array(path)
        if path.shape[0] == 0:
            continue
        # 缩放到mask尺寸
        xs = path[:,0] / env_dict["env_dims"][0] * (W-1)
        ys = path[:,1] / env_dict["env_dims"][1] * (H-1)
        ax.plot(xs, ys, color=color, linewidth=2)
# ---------------- 读取JSON ----------------
with open(JSON_FILE, "r") as f:
    env_list = json.load(f)
env_dict = np.random.choice(env_list)
mask = create_env_mask(env_dict)

# ---------------- 加载CAE ----------------
mask_size = IMG_SIZE[0]
encoder = AE.Encoder_CNN_2D(mask_size=mask_size)
dummy = torch.zeros(1,1,mask_size,mask_size)
with torch.no_grad():
    latent, enc_feats = encoder(dummy)
# enc_feats[-1] 就是最后的 conv 特征图
C,H,W = enc_feats[-1].shape[1:]
decoder = AE.Decoder_CNN_2D(latent_dim=latent.shape[1], feature_map_size=H)


encoder.load_state_dict(torch.load(os.path.join(MODEL_PATH,"encoder_best.pth"), map_location=DEVICE))
decoder.load_state_dict(torch.load(os.path.join(MODEL_PATH,"decoder_best.pth"), map_location=DEVICE))
encoder.to(DEVICE).eval()
decoder.to(DEVICE).eval()

# ---------------- 推理 ----------------
mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    latent, enc_feats = encoder(mask_tensor)
    recon = decoder(latent, enc_feats)

recon_np = recon.squeeze().cpu().numpy()
recon_sigmoid = torch.sigmoid(recon).squeeze().cpu().numpy()  # BCE输出

# ---------------- 可视化 ----------------
fig, axes = plt.subplots(1,3,figsize=(14,5))
axes[0].set_title("Original Environment")
draw_env(axes[0], env_dict)
draw_paths(axes[0], env_dict, img_size=env_dict["env_dims"], color='red')

axes[1].set_title("Mask Map")
axes[1].imshow(mask, cmap="gray")
draw_paths(axes[1], env_dict, img_size=IMG_SIZE, color='red')
axes[1].axis('on')

axes[2].set_title("CAE Reconstructed")
axes[2].imshow(recon_sigmoid, cmap="gray")
draw_paths(axes[2], env_dict, img_size=IMG_SIZE, color='red')
axes[2].axis('on')

plt.tight_layout()
plt.show()
