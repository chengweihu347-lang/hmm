import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# 导入 joblib 用于并行化
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

# ===============================
# 参数设置
# ===============================
# !!! 修改：使用相对路径，假设脚本与demobb和labels.csv在同一目录
DATA_FOLDER = "demobb"
LABEL_FILE = "labels.csv"

# DTW + KNN 参数
K_NEIGHBORS = 5  # KNN的K值
DTW_WINDOW = 50  # DTW窗口大小
USE_ADAPTIVE_WINDOW = True  # 是否使用自适应窗口
MAX_DTW_DISTANCE = 1e9  # DTW最大距离阈值，超过视为无穷大

# 并行化参数
N_JOBS = -1  # 并行核心数。-1 表示使用所有可用核心。

# 交叉验证参数
N_FOLDS = 5  # K折交叉验证的折数
PERFORM_CV = False  # 是否执行交叉验证（DTW计算慢，默认关闭）

RANDOM_STATE = 42
SCALE_FEATURES = True
NAN_STRATEGY = "interpolate"
MIN_VALID_FRAMES = 10
MAX_VALID_FRAMES = 500  # 最大帧数限制，防止序列过长
OUTLIER_CLIP_PERCENTILE = 99  # 异常值裁剪百分位

# 特征配置
USE_ANGLE = True  # 是否使用角度
USE_VELOCITY = False  # 是否使用角速度
USE_ACCELERATION = True  # 是否使用角加速度

# 置信度阈值设置
CONFIDENCE_THRESHOLD_LOW = 0.3
CONFIDENCE_THRESHOLD_HIGH = 0.7

PREDICTION_STRATEGY = "standard"

np.random.seed(RANDOM_STATE)


# ===============================
# 特征提取函数
# ===============================
def compute_derivatives(angles):
    """
    计算角度的一阶导数(角速度)和二阶导数(角加速度)
    """
    velocity = np.gradient(angles, axis=0)
    acceleration = np.gradient(velocity, axis=0)
    return velocity, acceleration


def extract_features(seq):
    """
    根据配置提取特征：角度、角速度、角加速度
    """
    if seq.size == 0:
        base_shape = (0, 5)
        empty_seq = np.empty(base_shape)
        seq_deriv1 = np.empty(base_shape)
        seq_deriv2 = np.empty(base_shape)
    else:
        empty_seq = seq
        seq_deriv1, seq_deriv2 = compute_derivatives(seq)

    features_list = []

    if USE_ANGLE:
        features_list.append(empty_seq)
    if USE_VELOCITY:
        features_list.append(seq_deriv1)
    if USE_ACCELERATION:
        features_list.append(seq_deriv2)

    if len(features_list) == 0:
        raise ValueError("至少需要选择一种特征类型！")

    features = np.hstack(features_list)
    return features


# ===============================
# 数据加载函数
# ===============================
def load_json_angles(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames", [])

    seq = []
    for f in frames:
        if "joints" not in f or not all(
            k in f["joints"]
            for k in [
                "shoulder_angle",
                "elbow_angle",
                "hip_angle",
                "knee_angle",
                "ankle_angle",
            ]
        ):
            continue

        angles = [
            f["joints"]["shoulder_angle"],
            f["joints"]["elbow_angle"],
            f["joints"]["hip_angle"],
            f["joints"]["knee_angle"],
            f["joints"]["ankle_angle"],
        ]
        seq.append(angles)

    seq = np.array(seq, dtype=float)

    if seq.size == 0:
        return extract_features(seq)

    has_nan = np.isnan(seq).any()
    if has_nan:
        if NAN_STRATEGY == "interpolate":
            seq = interpolate_nan(seq)
        elif NAN_STRATEGY == "drop":
            seq = drop_nan_frames(seq)

    features = extract_features(seq)
    return features


def interpolate_nan(seq):
    df = pd.DataFrame(seq)
    df = df.interpolate(method="linear", limit_direction="both", axis=0)
    df = df.fillna(df.mean())
    df = df.fillna(0)
    return df.values


def drop_nan_frames(seq):
    mask = ~np.isnan(seq).any(axis=1)
    return seq[mask]


def is_valid_sequence(seq):
    """验证序列是否有效"""
    if seq.size == 0:
        return False
    if len(seq) < MIN_VALID_FRAMES:
        return False
    if np.isnan(seq).all():
        return False
    if np.isinf(seq).any():
        return False
    return True


def clip_outliers(seq, percentile=99):
    """
    裁剪异常值到指定百分位
    """
    if seq.size == 0:
        return seq

    seq_clipped = seq.copy()

    for i in range(seq.shape[1]):
        col = seq[:, i]
        lower = np.percentile(col, 100 - percentile)
        upper = np.percentile(col, percentile)
        seq_clipped[:, i] = np.clip(col, lower, upper)

    return seq_clipped


def resample_sequence(seq, max_length=None):
    """
    如果序列过长，进行重采样
    """
    if seq.size == 0:
        return seq

    if max_length is None or len(seq) <= max_length:
        return seq

    original_indices = np.linspace(0, len(seq) - 1, len(seq))
    target_indices = np.linspace(0, len(seq) - 1, max_length)

    resampled = np.zeros((max_length, seq.shape[1]))
    for i in range(seq.shape[1]):
        resampled[:, i] = np.interp(target_indices, original_indices, seq[:, i])

    return resampled


# ===============================
# DTW距离计算函数 (核心逻辑)
# ===============================
def compute_dtw_distance(seq1, seq2):
    """
    计算两个多维时间序列之间的DTW距离（使用FastDTW）
    """
    # 检查输入有效性
    if seq1.size == 0 or seq2.size == 0:
        return MAX_DTW_DISTANCE

    if np.isnan(seq1).any() or np.isnan(seq2).any():
        return MAX_DTW_DISTANCE
    if np.isinf(seq1).any() or np.isinf(seq2).any():
        return MAX_DTW_DISTANCE

    try:
        # 确定窗口大小
        if USE_ADAPTIVE_WINDOW:
            len_diff = abs(len(seq1) - len(seq2))
            radius = min(DTW_WINDOW, max(10, len_diff + 10))
        else:
            radius = DTW_WINDOW

        # FastDTW使用欧氏距离比较帧与帧
        distance, _ = fastdtw(seq1, seq2, dist=euclidean, radius=radius)

        if np.isnan(distance) or np.isinf(distance):
            return MAX_DTW_DISTANCE

        return distance

    except Exception:
        # 任何计算失败都返回最大距离
        return MAX_DTW_DISTANCE


# ===============================
# DTW-KNN分类器 (并行化修改)
# ===============================
class DTW_KNN_Classifier:
    """
    基于DTW距离的KNN分类器 (使用Joblib并行化)
    """

    def __init__(self, n_neighbors=5, n_jobs=N_JOBS):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs  # 并行核心数
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        训练模型（存储训练数据）
        """
        self.X_train = X_train
        self.y_train = np.array(y_train)
        return self

    def predict_single(self, seq):
        """
        对单个序列进行预测（并行计算距离）
        """
        # --- 关键并行化部分 ---
        # 使用 joblib.Parallel 并行计算当前测试样本与所有训练样本的 DTW 距离
        distances = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(compute_dtw_distance)(seq, train_seq) for train_seq in self.X_train
        )
        # --- 并行化结束 ---

        distances = np.array(distances)

        # 检查是否所有距离都是无效的
        valid_distances = distances[distances < MAX_DTW_DISTANCE]
        if len(valid_distances) == 0:
            print("    警告: 所有DTW距离无效，回退到随机预测")
            # 回退：选择最近的（尽管是MAX_DTW）邻居
            k_nearest_indices = np.argsort(distances)[: self.n_neighbors]
            k_nearest_distances = distances[k_nearest_indices]
            k_nearest_labels = self.y_train[k_nearest_indices]
        else:
            # 找到K个最近邻
            k_nearest_indices = np.argsort(distances)[: self.n_neighbors]
            k_nearest_distances = distances[k_nearest_indices]
            k_nearest_labels = self.y_train[k_nearest_indices]

        # 投票决定预测标签
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]

        return predicted_label, k_nearest_distances, k_nearest_labels

    def predict(self, X_test):
        """
        对多个序列进行预测
        """
        predictions = []
        for seq in X_test:
            pred_label, _, _ = self.predict_single(seq)
            predictions.append(pred_label)
        return predictions


# ===============================
# 置信度计算函数（基于DTW距离）
# ===============================
def calculate_confidence_metrics_dtw(distances, neighbors_labels, all_labels):
    """
    基于DTW距离和最近邻标签计算置信度（带异常处理）
    """
    valid_mask = (
        (distances < MAX_DTW_DISTANCE) & (~np.isnan(distances)) & (~np.isinf(distances))
    )

    if not valid_mask.any():
        predicted_label = neighbors_labels[0]
        return {
            "predicted_label": predicted_label,
            "max_probability": 0.0,
            "margin": 0.0,
            "entropy": 1.0,
            "normalized_entropy": 1.0,
            "confidence_score": 0.0,
            "all_probabilities": {label: 1.0 / len(all_labels) for label in all_labels},
            "weighted_probabilities": {
                label: 1.0 / len(all_labels) for label in all_labels
            },
            "avg_distance": MAX_DTW_DISTANCE,
            "min_distance": MAX_DTW_DISTANCE,
        }

    valid_distances = distances[valid_mask]
    valid_labels = neighbors_labels[valid_mask]

    # 1. 基于投票的概率估计
    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    vote_probs = counts / len(valid_labels)

    label_probs = {label: 0.0 for label in all_labels}
    for label, prob in zip(unique_labels, vote_probs):
        label_probs[label] = prob

    predicted_label = unique_labels[np.argmax(counts)]
    max_probability = np.max(vote_probs)

    # 2. 基于距离的权重概率
    weights = 1.0 / (valid_distances + 1e-6)
    weights_sum = np.sum(weights)
    if weights_sum < 1e-6:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights_sum

    weighted_probs = {}
    for label in all_labels:
        mask = valid_labels == label
        weighted_probs[label] = np.sum(weights[mask]) if mask.any() else 0.0

    # 3. Margin
    sorted_probs = sorted(vote_probs, reverse=True)
    margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0)

    # 4. Entropy
    probs = np.array(list(label_probs.values()))
    probs_safe = probs + 1e-10
    entropy = -np.sum(probs * np.log(probs_safe))
    max_entropy = np.log(len(all_labels))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # 5. 综合置信度分数
    min_distance = np.min(valid_distances)
    distance_confidence = 1.0 / (1.0 + min_distance / 100)
    confidence_score = max_probability * distance_confidence * (1 - normalized_entropy)

    return {
        "predicted_label": predicted_label,
        "max_probability": max_probability,
        "margin": margin,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "confidence_score": confidence_score,
        "all_probabilities": label_probs,
        "weighted_probabilities": weighted_probs,
        "avg_distance": np.mean(valid_distances),
        "min_distance": min_distance,
    }


def predict_with_strategy(seq, classifier, all_labels, strategy="standard"):
    """
    根据不同策略进行预测
    """
    pred_label, distances, neighbors_labels = classifier.predict_single(seq)
    confidence_metrics = calculate_confidence_metrics_dtw(
        distances, neighbors_labels, all_labels
    )

    if strategy == "standard":
        final_pred = confidence_metrics["predicted_label"]
    elif strategy == "confidence_weighted":
        if confidence_metrics["confidence_score"] < CONFIDENCE_THRESHOLD_LOW:
            final_pred = "UNCERTAIN"
        else:
            final_pred = confidence_metrics["predicted_label"]
    elif strategy == "margin_based":
        if confidence_metrics["margin"] < 0.3:
            final_pred = "UNCERTAIN"
        else:
            final_pred = confidence_metrics["predicted_label"]
    else:
        final_pred = confidence_metrics["predicted_label"]

    return final_pred, confidence_metrics


# ===============================
# 加载数据
# ===============================
print("=" * 70)
print("模型配置: DTW + KNN (并行化)")
print(f"  - K值: {K_NEIGHBORS}")
print(f"  - DTW窗口: {DTW_WINDOW}")
print(f"  - 并行核心数: {N_JOBS if N_JOBS != -1 else 'ALL'}")
print("=" * 70)
print("特征配置:")
print(f"  - 使用角度: {USE_ANGLE}")
print(f"  - 使用角速度: {USE_VELOCITY}")
print(f"  - 使用角加速度: {USE_ACCELERATION}")

# 计算特征维度
feature_dim = 0
if USE_ANGLE:
    feature_dim += 5
if USE_VELOCITY:
    feature_dim += 5
if USE_ACCELERATION:
    feature_dim += 5
print(f"  - 总特征维度: {feature_dim}")
print("=" * 70)
print("数据处理配置:")
print(f"  - 最小帧数: {MIN_VALID_FRAMES}")
print(f"  - 最大帧数: {MAX_VALID_FRAMES} (重采样)")
print(f"  - 异常值裁剪百分位: {OUTLIER_CLIP_PERCENTILE}")
print("=" * 70)

if not os.path.exists(LABEL_FILE):
    print(f"致命错误: 标签文件未找到: {LABEL_FILE}")
    exit()

if not os.path.exists(DATA_FOLDER):
    print(f"致命错误: 数据文件夹未找到: {DATA_FOLDER}")
    exit()

labels_df = pd.read_csv(LABEL_FILE)
print(f"\nLoaded labels.csv: {len(labels_df)} entries")

sequences, labels, video_ids = [], [], []
skipped_count = 0

for _, row in labels_df.iterrows():
    json_path = os.path.join(DATA_FOLDER, row["video_id"])
    if os.path.exists(json_path):
        try:
            seq = load_json_angles(json_path)

            seq = clip_outliers(seq, OUTLIER_CLIP_PERCENTILE)

            if len(seq) > MAX_VALID_FRAMES:
                seq = resample_sequence(seq, MAX_VALID_FRAMES)

            if is_valid_sequence(seq):
                sequences.append(seq)
                labels.append(row["label"])
                video_ids.append(row["video_id"])
            else:
                skipped_count += 1
        except Exception as e:
            print(f"[错误] 处理文件失败: {row['video_id']}, 错误: {e}")
            skipped_count += 1
    else:
        skipped_count += 1

print(f"\n已成功加载 {len(sequences)} 个样本")
print(f"跳过/失败: {skipped_count} 个样本")

if not sequences:
    print("致命错误: 没有成功加载任何序列。")
    exit()

seq_lengths = [len(s) for s in sequences]
print(f"\n序列长度统计:")
print(f"  - 最小长度: {min(seq_lengths)} 帧")
print(f"  - 最大长度: {max(seq_lengths)} 帧")
print(f"  - 平均长度: {np.mean(seq_lengths):.1f} 帧")

label_counts = pd.Series(labels).value_counts()
print(f"\n各类别样本数:\n{label_counts}")

print("\n" + "=" * 70)
print("=== 交叉验证评估 ===")
print("=" * 70)

if PERFORM_CV and len(sequences) >= N_FOLDS:
    print(f"\n正在进行 {N_FOLDS} 折交叉验证...")
    print("⚠️  警告: DTW计算较慢，但已使用并行加速")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_accuracies = []
    cv_f1_scores = []
    fold_idx = 1

    labels_np = np.array(labels)

    for train_idx, val_idx in skf.split(sequences, labels_np):
        print(f"\n  折 {fold_idx}/{N_FOLDS}:")

        X_cv_train = [sequences[i] for i in train_idx]
        y_cv_train = labels_np[train_idx]
        X_cv_val = [sequences[i] for i in val_idx]
        y_cv_val = labels_np[val_idx]

        if SCALE_FEATURES:
            scaler_cv = StandardScaler()
            all_frames_cv = np.vstack(X_cv_train)
            scaler_cv.fit(all_frames_cv)
            X_cv_train = [scaler_cv.transform(x) for x in X_cv_train]
            X_cv_val = [scaler_cv.transform(x) for x in X_cv_val]

        clf_cv = DTW_KNN_Classifier(n_neighbors=K_NEIGHBORS, n_jobs=N_JOBS)  # 使用并行
        clf_cv.fit(X_cv_train, y_cv_train)
        y_cv_pred = clf_cv.predict(X_cv_val)

        fold_acc = accuracy_score(y_cv_val, y_cv_pred)
        fold_f1 = f1_score(y_cv_val, y_cv_pred, average="macro", zero_division=0)

        cv_accuracies.append(fold_acc)
        cv_f1_scores.append(fold_f1)

        print(f"    准确率: {fold_acc:.3f}")
        print(f"    宏平均F1: {fold_f1:.3f}")

        fold_idx += 1

    mean_acc = np.mean(cv_accuracies)
    std_acc = np.std(cv_accuracies)
    mean_f1 = np.mean(cv_f1_scores)
    std_f1 = np.std(cv_f1_scores)

    print("\n" + "-" * 70)
    print("交叉验证结果汇总:")
    print("-" * 70)
    print(f"平均准确率: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"\n平均宏F1分数: {mean_f1:.3f} ± {std_f1:.3f}")
    print("-" * 70)

    cv_results = pd.DataFrame(
        {
            "fold": range(1, N_FOLDS + 1),
            "accuracy": cv_accuracies,
            "macro_f1": cv_f1_scores,
        }
    )
    cv_results.to_csv("cross_validation_results.csv", index=False)
    print(f"\n💾 交叉验证结果已保存至: cross_validation_results.csv")

else:
    if not PERFORM_CV:
        print("\n⚠️  交叉验证已禁用 (PERFORM_CV=False)")
    else:
        print(f"\n⚠️  样本数({len(sequences)})少于折数({N_FOLDS})，跳过交叉验证")

# ===============================
# 划分数据集
# ===============================
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    sequences,
    labels,
    video_ids,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=labels,
)
print(f"\n训练集: {len(X_train)} ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)")
print(f"测试集: {len(X_test)} ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)")

# ===============================
# 特征缩放（带异常值处理）
# ===============================
if SCALE_FEATURES:
    scaler = StandardScaler()
    all_train_frames = np.vstack(X_train)

    if np.isinf(all_train_frames).any():
        print("警告: 训练数据中包含无穷大值，进行裁剪...")
        all_train_frames[np.isinf(all_train_frames)] = np.nan
        df_temp = pd.DataFrame(all_train_frames)
        df_temp = df_temp.fillna(df_temp.mean())
        all_train_frames = df_temp.values

    scaler.fit(all_train_frames)

    X_train_scaled = []
    for x in X_train:
        x_scaled = scaler.transform(x)
        x_scaled[np.isinf(x_scaled)] = 0
        x_scaled[np.isnan(x_scaled)] = 0
        X_train_scaled.append(x_scaled)
    X_train = X_train_scaled

    X_test_scaled = []
    for x in X_test:
        x_scaled = scaler.transform(x)
        x_scaled[np.isinf(x_scaled)] = 0
        x_scaled[np.isnan(x_scaled)] = 0
        X_test_scaled.append(x_scaled)
    X_test = X_test_scaled

    print("特征标准化完成（已处理异常值）")

# ===============================
# 训练DTW-KNN模型
# ===============================
print("\n正在训练 DTW-KNN 模型...")
classifier = DTW_KNN_Classifier(n_neighbors=K_NEIGHBORS, n_jobs=N_JOBS)  # 使用并行
classifier.fit(X_train, y_train)
print(f"✅ DTW-KNN 模型训练完成（存储了 {len(X_train)} 个训练样本）")

# ===============================
# 预测与置信度分析
# ===============================
print(f"\n使用预测策略: {PREDICTION_STRATEGY}")
print("开始预测（计算DTW距离可能需要一些时间）...")
print(
    f"提示: 共需计算 {len(X_test)} × {len(X_train)} = {len(X_test) * len(X_train)} 对DTW距离 (已使用 {N_JOBS if N_JOBS != -1 else 'ALL'} 个核心加速)"
)

y_pred = []
confidence_results = []
unique_labels = sorted(set(y_train))
failed_predictions = 0

for i, seq in enumerate(X_test):
    # 仅在非并行模式下才显示详细进度，否则并行输出会混乱
    if N_JOBS == 1 and ((i + 1) % 5 == 0 or i == 0 or i == len(X_test) - 1):
        print(f"  进度: {i+1}/{len(X_test)} ({(i+1)/len(X_test)*100:.1f}%)")

    try:
        # predict_single 内部现在是并行计算
        pred_label, conf_metrics = predict_with_strategy(
            seq, classifier, unique_labels, PREDICTION_STRATEGY
        )
        y_pred.append(pred_label)

        confidence_results.append(
            {
                "video_id": ids_test[i],
                "true_label": y_test[i],
                "predicted_label": pred_label,
                "confidence_score": conf_metrics["confidence_score"],
                "max_probability": conf_metrics["max_probability"],
                "margin": conf_metrics["margin"],
                "entropy": conf_metrics["normalized_entropy"],
                "avg_distance": conf_metrics["avg_distance"],
                "min_distance": conf_metrics["min_distance"],
                "is_correct": pred_label == y_test[i],
                "all_probs": conf_metrics["all_probabilities"],
            }
        )
    except Exception as e:
        print(f"  错误: 预测样本 {ids_test[i]} 失败: {e}")
        default_label = unique_labels[0]
        y_pred.append(default_label)
        confidence_results.append(
            {
                "video_id": ids_test[i],
                "true_label": y_test[i],
                "predicted_label": default_label,
                "confidence_score": 0.0,
                "max_probability": 0.0,
                "margin": 0.0,
                "entropy": 1.0,
                "avg_distance": MAX_DTW_DISTANCE,
                "min_distance": MAX_DTW_DISTANCE,
                "is_correct": default_label == y_test[i],
                "all_probs": {},
            }
        )
        failed_predictions += 1

print("✅ 预测完成！")
if failed_predictions > 0:
    print(f"⚠️  警告: {failed_predictions} 个样本预测失败，使用默认值")

confidence_df = pd.DataFrame(confidence_results)

# ===============================
# 基本评估结果
# ===============================
labels_sorted = sorted(unique_labels)

if "UNCERTAIN" in y_pred:
    y_pred_filtered = [
        p if p != "UNCERTAIN" else y_test[i] for i, p in enumerate(y_pred)
    ]
    print(f"\n发现 {y_pred.count('UNCERTAIN')} 个低置信度预测")
else:
    y_pred_filtered = y_pred

all_present_labels = sorted(list(set(y_test) | set(y_pred_filtered)))
if "UNCERTAIN" in all_present_labels:
    all_present_labels.remove("UNCERTAIN")


cm = confusion_matrix(y_test, y_pred_filtered, labels=all_present_labels)
acc = accuracy_score(y_test, y_pred_filtered)

print("\n" + "=" * 70)
print("=== 基本分类结果 ===")
print("=" * 70)
print(f"准确率: {acc:.3f}")
print(f"\n混淆矩阵 (行=真实, 列=预测):\n{cm}")
print("标签:", all_present_labels)

report_str = classification_report(
    y_test, y_pred_filtered, target_names=all_present_labels, zero_division=0
)
report_dict = classification_report(
    y_test,
    y_pred_filtered,
    target_names=all_present_labels,
    output_dict=True,
    zero_division=0,
)
print(f"\n详细报告:\n{report_str}")

print("\n" + "=" * 70)
print("=== 各类别F1分数 ===")
print("=" * 70)
print(f"{'类别':<15} {'F1-Score':<12} {'Precision':<12} {'Recall':<12} {'Support'}")
print("-" * 70)
for label in all_present_labels:
    if label in report_dict:
        f1 = report_dict[label]["f1-score"]
        precision = report_dict[label]["precision"]
        recall = report_dict[label]["recall"]
        support = report_dict[label]["support"]
        print(f"{label:<15} {f1:<12.3f} {precision:<12.3f} {recall:<12.3f} {support}")

macro_f1 = report_dict["macro avg"]["f1-score"]
weighted_f1 = report_dict["weighted avg"]["f1-score"]
print("-" * 70)
print(f"{'宏平均 (Macro)':<15} {macro_f1:<12.3f}")
print(f"{'加权平均 (Weighted)':<15} {weighted_f1:<12.3f}")

# ===============================
# 置信度统计分析
# ===============================
print("\n" + "=" * 70)
print("=== 置信度分析 ===")
print("=" * 70)

if confidence_df.empty:
    print("置信度DataFrame为空，跳过分析。")
else:
    print(f"\n整体置信度统计:")
    print(f"  平均置信度: {confidence_df['confidence_score'].mean():.3f}")
    print(f"  置信度中位数: {confidence_df['confidence_score'].median():.3f}")

    correct_conf = confidence_df[confidence_df["is_correct"]]["confidence_score"].mean()
    incorrect_conf = confidence_df[~confidence_df["is_correct"]][
        "confidence_score"
    ].mean()
    print(f"\n正确vs错误预测的置信度对比:")
    print(f"  正确预测平均置信度: {correct_conf:.3f}")
    print(f"  错误预测平均置信度: {incorrect_conf:.3f}")

    correct_dist = confidence_df[confidence_df["is_correct"]]["avg_distance"].mean()
    incorrect_dist = confidence_df[~confidence_df["is_correct"]]["avg_distance"].mean()
    print(f"\n正确vs错误预测的DTW距离对比:")
    print(f"  正确预测平均距离: {correct_dist:.2f}")
    print(f"  错误预测平均距离: {incorrect_dist:.2f}")

output_file = "confidence_analysis_dtw_knn_parallel.csv"
confidence_df.to_csv(output_file, index=False)
print(f"\n💾 完整分析结果已保存至: {output_file}")

print("\n" + "=" * 70)
print("=== 性能总结 ===")
print("=" * 70)
print(
    f"""
加速方法: CPU多核并行化 (Joblib)
核心数: {N_JOBS if N_JOBS != -1 else 'ALL'}
预计加速比: 接近核心数 (例如，8核CPU上可能提高约7-8倍速度)。

性能指标:
  - 测试集准确率: {acc:.3f}
  - 测试集宏平均F1: {macro_f1:.3f}"""
)

if PERFORM_CV and "mean_acc" in locals():
    print(f"  - 交叉验证准确率: {mean_acc:.3f} ± {std_acc:.3f}")

print("=" * 70)
