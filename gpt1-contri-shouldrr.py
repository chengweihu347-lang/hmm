"""
compare_motion_visuals_by_labelcsv_safe.py

修正版：处理 JSON 中的 None 值（转换为 np.nan 并插值），并对短序列或无效序列做过滤。
实现三种可视化：
 - 方法5：平均曲线 + 标准差阴影（肩关节为示例）
 - 方法6：箱线图（平均与标准差）
 - 方法8：t-SNE（所有有效帧）
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.ndimage import uniform_filter1d

# 设置华文宋体
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",  # macOS默认中文字体
    "Hiragino Sans GB",  # 另一个macOS中文字体
    "Microsoft YaHei",  # 如果安装了微软雅黑
    "SimHei",  # 黑体
]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["font.size"] = 12  # 设置默认字体大小
plt.rcParams["axes.labelsize"] = 12  # 坐标轴标签大小
plt.rcParams["axes.titlesize"] = 14  # 标题大小
plt.rcParams["xtick.labelsize"] = 10  # x轴刻度标签大小
plt.rcParams["ytick.labelsize"] = 10  # y轴刻度标签大小
plt.rcParams["legend.fontsize"] = 10  # 图例字体大小
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
# ---------- 配置 ----------

data_dir = os.path.join("demobb")
label_path = os.path.join("labels.csv")

MIN_VALID_FRAMES = 10  # 序列至少应有多少有效帧
MAX_PLOT_LEN = 200  # 平均曲线截断到的最大帧数（避免极长序列）
TSNE_PERPLEXITY = 30


# ---------- 工具函数 ----------
def safe_get_angle(frame, key):
    """从frame里安全拿角度值，将 None 转为 np.nan"""
    v = frame.get("joints", {}).get(key, None)
    if v is None:
        return np.nan
    try:
        return float(v)
    except:
        return np.nan


def load_and_clean_json(json_path):
    """
    读取 json -> 原始角度数组（N x 5），将 None -> np.nan，然后插值填充。
    返回 cleaned ndarray 或 None（如果无效或太短）。
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    if len(frames) == 0:
        return None

    # 提取为 N x 5，None -> np.nan
    arr = np.array(
        [
            [
                safe_get_angle(fr, "shoulder_angle"),
                safe_get_angle(fr, "elbow_angle"),
                safe_get_angle(fr, "hip_angle"),
                safe_get_angle(fr, "knee_angle"),
                safe_get_angle(fr, "ankle_angle"),
            ]
            for fr in frames
        ],
        dtype=float,
    )

    # 如果整张表全是 NaN，直接丢弃
    if np.isnan(arr).all():
        return None

    # 对每一列进行插值填充（先线性插值，再前向/后向填充，再列均值）
    df = pd.DataFrame(arr, columns=["shoulder", "elbow", "hip", "knee", "ankle"])
    df_interp = df.interpolate(method="linear", limit_direction="both", axis=0)
    df_interp = df_interp.fillna(df_interp.mean())  # 列均值填充剩余 NaN
    df_interp = df_interp.fillna(0.0)  # 极端情况下再用 0 填充

    cleaned = df_interp.values

    # 再次检查有效帧（不完全是 NaN）
    if np.isnan(cleaned).all() or cleaned.shape[0] < MIN_VALID_FRAMES:
        return None

    return cleaned


# ---------- 读取标签表 ----------
if not os.path.exists(label_path):
    raise FileNotFoundError(f"标签文件未找到: {label_path}")

label_df = pd.read_csv(label_path)
label_map = dict(zip(label_df["video_id"], label_df["label"]))
print(f"Loaded label.csv with {len(label_map)} entries")

# ---------- 读取并清洗所有 JSON ----------
sequences = []
labels = []
skipped = []

for file in sorted(os.listdir(data_dir)):
    if not file.endswith(".json"):
        continue
    if file not in label_map:
        # 如果 label.csv 没有该文件则跳过（避免误读）
        print(f"[跳过] {file} 未在 label.csv 中列出")
        skipped.append(file)
        continue
    path = os.path.join(data_dir, file)
    try:
        seq = load_and_clean_json(path)
        if seq is None:
            print(f"[无效] {file} -> 清洗后无有效帧或帧数 < {MIN_VALID_FRAMES}")
            skipped.append(file)
            continue
        sequences.append(seq)
        labels.append(label_map[file])
    except Exception as e:
        print(f"[错误] 读取 {file} 时出错: {e}")
        skipped.append(file)

print(f"\n✅ 成功加载并清洗 {len(sequences)} 条序列；跳过 {len(skipped)} 条")

if len(sequences) == 0:
    raise RuntimeError(
        "没有可用的序列用于可视化（全部被跳过）。请检查数据或放宽 MIN_VALID_FRAMES 等条件。"
    )

# ---------- 方法5：平均曲线 + 标准差阴影（以肩角度为例） ----------
plt.figure(figsize=(10, 6))
unique_labels = sorted(set(labels))

# 先计算每类最小长度（但不超过 MAX_PLOT_LEN）
per_label_minlen = {}
for lab in unique_labels:
    lens = [len(s) for s, l in zip(sequences, labels) if l == lab]
    if len(lens) == 0:
        per_label_minlen[lab] = 0
    else:
        per_label_minlen[lab] = min(min(lens), MAX_PLOT_LEN)

# 为可比性，统一取所有类别的最小长度（如果想分别画也可以）
global_min_len = min([v for v in per_label_minlen.values() if v > 0])
if global_min_len < MIN_VALID_FRAMES:
    raise RuntimeError(
        "清洗后某些类别长度过短，无法比较。请降低 MIN_VALID_FRAMES 或检查数据。"
    )

print(f"Using frame length = {global_min_len} for averaging plots")

for label in unique_labels:
    seqs_lab = [s for s, l in zip(sequences, labels) if l == label]
    if len(seqs_lab) == 0:
        continue
    # 取肩角并截断到 global_min_len
    shoulder_mat = np.vstack([s[:global_min_len, 0] for s in seqs_lab])
    mean_curve = np.mean(shoulder_mat, axis=0)
    std_curve = np.std(shoulder_mat, axis=0)
    smooth_mean = uniform_filter1d(mean_curve, size=5)
    plt.plot(smooth_mean, label=f"{label} (n={len(seqs_lab)})")
    plt.fill_between(
        range(global_min_len),
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.15,
    )

plt.title("方法5：各类别平均肩角度（截断并插值后的结果）")
plt.xlabel("帧索引 (截断到相同长度)")
plt.ylabel("肩角度 (°)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- 方法6：箱线图（每条序列的平均与标准差） ----------
shoulder_mean = [np.mean(s[:, 0]) for s in sequences]
shoulder_std = [np.std(s[:, 0]) for s in sequences]
df_stats = pd.DataFrame(
    {"mean_angle": shoulder_mean, "std_angle": shoulder_std, "label": labels}
)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x="label", y="mean_angle", data=df_stats)
plt.title("方法6：每序列平均肩角度箱线图")
plt.subplot(1, 2, 2)
sns.boxplot(x="label", y="std_angle", data=df_stats)
plt.title("方法6：每序列肩角度标准差箱线图")
plt.tight_layout()
plt.show()

# ---------- 方法8：t-SNE（使用所有清洗后的帧） ----------
# 准备 X_all, y_all：按帧展开
X_all = np.vstack(sequences)  # M x 5
y_all = np.concatenate([[lab] * len(s) for s, lab in zip(sequences, labels)])

print(f"t-SNE input: {X_all.shape[0]} frames, {X_all.shape[1]} features")

# 可选：对 X_all 做标准化（使 t-SNE 更稳定）
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# 运行 t-SNE（注意大数据集可能很慢）
tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=42, init="random")
X_emb = tsne.fit_transform(X_all_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_emb[:, 0],
    y=X_emb[:, 1],
    hue=y_all,
    palette="tab10",
    alpha=0.6,
    s=8,
    linewidth=0,
)
plt.title("方法8：t-SNE (按帧) - 各类别分布")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.legend(title="label", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

print("可视化完成。")
