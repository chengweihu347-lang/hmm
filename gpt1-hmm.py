import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from hmmlearn.hmm import GMMHMM
# 可修改的值：N_COMPONENTS、N_MIXTURES（高斯混合状态）、MAX_ITER（iteration）、COV_TYPE、RANDOM_STATE、SCALE_FEATURES、NAN_STRATEGY、MIN_VALID_FRAMES     数据增强,修改特征标准化的方法，交叉验证
# ===============================
# 参数设置 0.5的准确率；    检查数据质量！数据质量分析，了解各类别的特征分布！！！！；
# 尝试不同的特征组合，特别是对jg类别；
# 分析预测置信度，找出低置信度的样本          优化特征与模型参数
# ===============================
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "demobb")
LABEL_FILE = os.path.join(os.path.dirname(__file__), "labels.csv")

N_COMPONENTS = 5  # 隐状态数，可调 (3~6)
N_MIXTURES = 3  # GMM混合数
MAX_ITER = 100  # EM迭代次数
COV_TYPE = "diag"
RANDOM_STATE = 42
SCALE_FEATURES = True

# NaN处理策略: 'interpolate' 或 'drop'
NAN_STRATEGY = "interpolate"  # 推荐使用插值
MIN_VALID_FRAMES = 10  # 最少有效帧数，少于此数的序列会被丢弃

np.random.seed(RANDOM_STATE)


# ===============================
# 1. 加载角度序列（处理NaN）
# ===============================
def load_json_angles(json_path):
    """
    加载JSON文件中的角度数据，处理NaN值
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    frames = data["frames"]

    # 提取角度数据
    seq = []
    for f in frames:
        angles = [
            f["joints"]["shoulder_angle"],
            f["joints"]["elbow_angle"],
            f["joints"]["hip_angle"],
            f["joints"]["knee_angle"],
            f["joints"]["ankle_angle"],
        ]
        seq.append(angles)

    seq = np.array(seq, dtype=float)

    #     # 添加角度的平方作为特征 # 添加ankle角度的变化率作为新特征
    # angles_squared = angles ** 2
    # seq.append(np.concatenate([angles, angles_squared]))
    # return np.array(seq)

    # 角度权重增加
    if labels == "jg":
        weights = [1.0, 1.0, 1.0, 1.0, 3.0]  # 对jg类别，ankle角度权重加倍
    else:
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # 其他类别保持原始权重
    # 应用权重
    seq = seq * weights

    # 检测NaN
    has_nan = np.isnan(seq).any()
    if has_nan:
        if NAN_STRATEGY == "interpolate":
            seq = interpolate_nan(seq)
        elif NAN_STRATEGY == "drop":
            seq = drop_nan_frames(seq)
        else:
            raise ValueError(f"未知的NaN处理策略: {NAN_STRATEGY}")

    return seq


def interpolate_nan(seq):
    """
    使用线性插值填充NaN值
    """
    df = pd.DataFrame(seq)

    # 先进行前向填充，再进行后向填充（处理开头和结尾的NaN）
    df = df.interpolate(method="linear", limit_direction="both", axis=0)

    # 如果还有NaN（比如整列都是NaN），用列均值填充
    df = df.fillna(df.mean())

    # 如果还有NaN（整列都是NaN且无法计算均值），用0填充
    df = df.fillna(0)

    return df.values


def drop_nan_frames(seq):
    """
    删除包含NaN的帧
    """
    mask = ~np.isnan(seq).any(axis=1)
    return seq[mask]


def is_valid_sequence(seq):
    """
    检查序列是否有效（帧数足够且无全NaN）
    """
    if len(seq) < MIN_VALID_FRAMES:
        return False
    if np.isnan(seq).all():
        return False
    return True


# ===============================
# 2. 加载标签和数据
# ===============================
if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError(f"标签文件未找到: {LABEL_FILE}")

labels_df = pd.read_csv(LABEL_FILE)
print(f"Loaded labels.csv: {len(labels_df)} entries")

# 对应每个json文件载入
sequences, labels = [], []
skipped_count = 0

for _, row in labels_df.iterrows():
    json_path = os.path.join(DATA_FOLDER, row["video_id"])
    if os.path.exists(json_path):
        try:
            seq = load_json_angles(json_path)

            # 验证序列有效性
            if is_valid_sequence(seq):
                sequences.append(seq)
                labels.append(row["label"])
                # 添加数据增强
            # if len(seq) > 20:  # 只对足够长的序列进行增强
            #     # 添加时间反转的序列
            #     sequences.append(seq[::-1])
            #     labels.append(row["label"])
            else:
                print(
                    f"[警告] 序列无效(帧数不足或全NaN): {row['video_id']}, 帧数: {len(seq)}"
                )
                skipped_count += 1
        except Exception as e:
            print(f"[错误] 处理文件失败: {row['video_id']}, 错误: {e}")
            skipped_count += 1
    else:
        print(f"[警告] 找不到文件: {row['video_id']}")
        skipped_count += 1

print(f"\n已成功加载 {len(sequences)} 个样本")
print(f"跳过/失败: {skipped_count} 个样本")

# 检查是否有足够的数据
if len(sequences) < 10:
    raise ValueError(f"有效样本数量太少 ({len(sequences)})，无法进行训练")

# 检查每个类别的样本数
label_counts = pd.Series(labels).value_counts()
print(f"\n各类别样本数:\n{label_counts}")

# ===============================
# 3. 划分训练集 / 测试集
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
)
print(f"\n训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ===============================
# 4. 特征缩放
# ===============================
if SCALE_FEATURES:
    scaler = StandardScaler()
    all_train_frames = np.vstack(X_train)
    scaler.fit(all_train_frames)
    X_train = [scaler.transform(x) for x in X_train]
    X_test = [scaler.transform(x) for x in X_test]
    print("特征标准化完成")

    # 添加特征工程
    # def add_features(seq):
    #     # 添加一阶差分
    #     diff = np.diff(seq, axis=0)
    #     # 添加二阶差分
    #     diff2 = np.diff(diff, axis=0)
    #     # 合并原始特征和差分特征
    #     return np.hstack([seq[1:], diff])

    # X_train = [add_features(x) for x in X_train]
    # X_test = [add_features(x) for x in X_test]
# ===============================
# 在模型训练前添加交叉验证
# from sklearn.model_selection import StratifiedKFold

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# cv_scores = []
# for train_idx, val_idx in skf.split(X_train, y_train):
#     X_train_fold = [X_train[i] for i in train_idx]
#     y_train_fold = [y_train[i] for i in train_idx]
#     X_val_fold = [X_train[i] for i in val_idx]
#     y_val_fold = [y_train[i] for i in val_idx]

# 训练和评估代码...

# 5. 训练每类 HMM 模型
# ===============================
models = {}
unique_labels = sorted(set(y_train))

for label in unique_labels:
    label_seqs = [X_train[i] for i in range(len(X_train)) if y_train[i] == label]
    X_concat = np.vstack(label_seqs)
    lengths = [len(x) for x in label_seqs]

    print(f"\n正在训练 HMM 模型: {label} ({len(label_seqs)} 条序列)...")
    model = GMMHMM(
        n_components=N_COMPONENTS,
        n_mix=N_MIXTURES,
        covariance_type=COV_TYPE,
        n_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.fit(X_concat, lengths)
    models[label] = model
    print(f"✅ 完成 {label} 模型训练")

# ===============================
# 6. 测试与分类
# ===============================
y_pred = []
for seq in X_test:
    logL = {label: models[label].score(seq) for label in models}
    pred_label = max(logL, key=logL.get)
    y_pred.append(pred_label)

# ===============================
# 7. 输出评估结果
# ===============================
labels_sorted = sorted(unique_labels)
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("=== 分类结果 ===")
print("=" * 50)
print(f"准确率: {acc:.3f}")
print(f"标签集: {labels_sorted}")
print(f"\n混淆矩阵:\n{cm}")
print(
    f"\n详细报告:\n{classification_report(y_test, y_pred, target_names=labels_sorted)}"
)

print("\n样例预测:")
for i in range(min(5, len(y_test))):
    print(f"  真实: {y_test[i]:<15} 预测: {y_pred[i]}")

# ===============================
# 8. 打印每类模型核心参数
# ===============================
print("\n" + "=" * 50)
print("=== 模型参数 ===")
print("=" * 50)
for lbl, model in models.items():
    print(f"\n--- {lbl} ---")
    print("初始状态分布 π:")
    print(np.round(model.startprob_, 3))
    print("\n状态转移矩阵 A:")
    print(np.round(model.transmat_, 3))
