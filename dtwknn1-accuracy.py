import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from dtaidistance import dtw
import warnings

warnings.filterwarnings("ignore")

# ===============================
# 参数设置    交叉验证已禁用 (PERFORM_CV=False)
# ===============================
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "demobb")
LABEL_FILE = os.path.join(os.path.dirname(__file__), "labels.csv")

# DTW + KNN 参数
K_NEIGHBORS = 5  # KNN的K值
DTW_WINDOW = 50  # DTW窗口大小（自适应调整，防止无穷大）
DTW_PSI = 2  # DTW的psi参数（端点松弛，降低以提高稳定性）
USE_ADAPTIVE_WINDOW = True  # 是否使用自适应窗口
MAX_DTW_DISTANCE = 1e6  # DTW最大距离阈值，超过视为无穷大

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
USE_ACCELERATION = False  # 是否使用角加速度

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

    Args:
        angles: shape (n_frames, n_joints) 的角度序列

    Returns:
        velocity: 角速度
        acceleration: 角加速度
    """
    velocity = np.gradient(angles, axis=0)
    acceleration = np.gradient(velocity, axis=0)

    return velocity, acceleration


def extract_features(seq):
    """
    根据配置提取特征：角度、角速度、角加速度

    Args:
        seq: shape (n_frames, 5) 的角度序列

    Returns:
        features: 组合后的特征矩阵
    """
    features_list = []

    if USE_ANGLE:
        features_list.append(seq)

    if USE_VELOCITY:
        velocity, _ = compute_derivatives(seq)
        features_list.append(velocity)

    if USE_ACCELERATION:
        _, acceleration = compute_derivatives(seq)
        features_list.append(acceleration)

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
    frames = data["frames"]

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
    if len(seq) < MIN_VALID_FRAMES:
        return False
    if len(seq) > MAX_VALID_FRAMES:
        return False
    if np.isnan(seq).all():
        return False
    if np.isinf(seq).any():  # 检查无穷大值
        return False
    return True


def clip_outliers(seq, percentile=99):
    """
    裁剪异常值到指定百分位

    Args:
        seq: shape (n_frames, n_features)
        percentile: 裁剪百分位数

    Returns:
        clipped_seq: 裁剪后的序列
    """
    seq_clipped = seq.copy()

    for i in range(seq.shape[1]):
        col = seq[:, i]
        # 计算百分位数
        lower = np.percentile(col, 100 - percentile)
        upper = np.percentile(col, percentile)
        # 裁剪
        seq_clipped[:, i] = np.clip(col, lower, upper)

    return seq_clipped


def resample_sequence(seq, max_length=None):
    """
    如果序列过长，进行重采样

    Args:
        seq: shape (n_frames, n_features)
        max_length: 最大长度

    Returns:
        resampled_seq: 重采样后的序列
    """
    if max_length is None or len(seq) <= max_length:
        return seq

    # 使用线性插值重采样
    original_indices = np.linspace(0, len(seq) - 1, len(seq))
    target_indices = np.linspace(0, len(seq) - 1, max_length)

    resampled = np.zeros((max_length, seq.shape[1]))
    for i in range(seq.shape[1]):
        resampled[:, i] = np.interp(target_indices, original_indices, seq[:, i])

    return resampled


# ===============================
# DTW距离计算函数
# ===============================
def compute_dtw_distance(seq1, seq2):
    """
    计算两个时间序列之间的DTW距离（带异常处理）

    Args:
        seq1: shape (n_frames1, n_features)
        seq2: shape (n_frames2, n_features)

    Returns:
        distance: DTW距离
    """
    # 检查输入有效性
    if np.isnan(seq1).any() or np.isnan(seq2).any():
        return MAX_DTW_DISTANCE

    if np.isinf(seq1).any() or np.isinf(seq2).any():
        return MAX_DTW_DISTANCE

    # 计算自适应窗口大小
    if USE_ADAPTIVE_WINDOW:
        # 窗口大小为两个序列长度差的绝对值 + 一个缓冲
        len_diff = abs(len(seq1) - len(seq2))
        adaptive_window = min(DTW_WINDOW, max(10, len_diff + 10))
    else:
        adaptive_window = DTW_WINDOW

    total_distance = 0.0
    n_features = seq1.shape[1]
    valid_features = 0

    for i in range(n_features):
        try:
            s1 = seq1[:, i].astype(np.float64)
            s2 = seq2[:, i].astype(np.float64)

            # 检查是否包含NaN或Inf
            if np.isnan(s1).any() or np.isnan(s2).any():
                continue
            if np.isinf(s1).any() or np.isinf(s2).any():
                continue

            # 使用dtaidistance库计算DTW距离
            distance = dtw.distance(s1, s2, window=adaptive_window, psi=DTW_PSI)

            # 检查结果是否有效
            if np.isnan(distance) or np.isinf(distance):
                distance = MAX_DTW_DISTANCE / n_features

            # 限制单个特征的最大距离
            distance = min(distance, MAX_DTW_DISTANCE / n_features)

            total_distance += distance
            valid_features += 1

        except Exception as e:
            # 如果计算失败，使用最大距离
            print(f"    警告: 特征{i}的DTW计算失败: {e}")
            total_distance += MAX_DTW_DISTANCE / n_features
            valid_features += 1

    # 返回平均距离
    if valid_features == 0:
        return MAX_DTW_DISTANCE

    avg_distance = total_distance / valid_features

    # 最终检查
    if np.isnan(avg_distance) or np.isinf(avg_distance):
        return MAX_DTW_DISTANCE

    return avg_distance


# ===============================
# 自定义DTW-KNN分类器
# ===============================
class DTW_KNN_Classifier:
    """
    基于DTW距离的KNN分类器
    """

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        训练模型（实际上只是存储训练数据）

        Args:
            X_train: list of arrays, 训练序列
            y_train: list, 训练标签
        """
        self.X_train = X_train
        self.y_train = np.array(y_train)
        return self

    def predict_single(self, seq):
        """
        对单个序列进行预测

        Args:
            seq: shape (n_frames, n_features)

        Returns:
            predicted_label: 预测标签
            distances: K个最近邻的距离
            neighbors_labels: K个最近邻的标签
        """
        # 计算测试样本与所有训练样本的DTW距离
        distances = []
        for train_seq in self.X_train:
            dist = compute_dtw_distance(seq, train_seq)
            distances.append(dist)

        distances = np.array(distances)

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

        Args:
            X_test: list of arrays

        Returns:
            predictions: list of predicted labels
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
    基于DTW距离和最近邻标签计算置信度

    Args:
        distances: K个最近邻的距离
        neighbors_labels: K个最近邻的标签
        all_labels: 所有可能的标签

    Returns:
        dict: 包含置信度指标
    """
    # 1. 基于投票的概率估计
    unique_labels, counts = np.unique(neighbors_labels, return_counts=True)
    vote_probs = counts / len(neighbors_labels)

    # 构建所有标签的概率字典
    label_probs = {label: 0.0 for label in all_labels}
    for label, prob in zip(unique_labels, vote_probs):
        label_probs[label] = prob

    predicted_label = unique_labels[np.argmax(counts)]
    max_probability = np.max(vote_probs)

    # 2. 基于距离的权重概率（距离越小，权重越大）
    # 使用高斯核函数
    weights = np.exp(-distances / np.mean(distances))
    weights = weights / np.sum(weights)

    weighted_probs = {}
    for label in all_labels:
        mask = neighbors_labels == label
        weighted_probs[label] = np.sum(weights[mask])

    # 3. Margin (最高概率与第二高概率的差距)
    sorted_probs = sorted(vote_probs, reverse=True)
    margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0)

    # 4. Entropy
    probs = np.array(list(label_probs.values()))
    probs_safe = probs + 1e-10
    entropy = -np.sum(probs * np.log(probs_safe))
    max_entropy = np.log(len(all_labels))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # 5. 综合置信度分数
    # 考虑投票一致性和距离
    avg_distance = np.mean(distances)
    distance_confidence = 1.0 / (1.0 + avg_distance / 100)  # 归一化距离
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
        "avg_distance": avg_distance,
        "min_distance": np.min(distances),
    }


def predict_with_strategy(seq, classifier, all_labels, strategy="standard"):
    """
    根据不同策略进行预测

    Args:
        seq: 输入序列
        classifier: DTW-KNN分类器
        all_labels: 所有可能的标签列表
        strategy: 预测策略

    Returns:
        predicted_label, confidence_metrics
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
        if confidence_metrics["margin"] < 0.3:  # 调整阈值
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
print("模型配置: DTW + KNN")
print(f"  - K值: {K_NEIGHBORS}")
print(f"  - DTW窗口: {DTW_WINDOW if DTW_WINDOW else '无限制'}")
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

if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError(f"标签文件未找到: {LABEL_FILE}")

labels_df = pd.read_csv(LABEL_FILE)
print(f"\nLoaded labels.csv: {len(labels_df)} entries")

sequences, labels, video_ids = [], [], []
skipped_count = 0

for _, row in labels_df.iterrows():
    json_path = os.path.join(DATA_FOLDER, row["video_id"])
    if os.path.exists(json_path):
        try:
            seq = load_json_angles(json_path)
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

label_counts = pd.Series(labels).value_counts()
print(f"\n各类别样本数:\n{label_counts}")

print("\n" + "=" * 70)
print("=== 交叉验证评估 ===")
print("=" * 70)

if PERFORM_CV and len(sequences) >= N_FOLDS:
    print(f"\n正在进行 {N_FOLDS} 折交叉验证...")
    print("⚠️  警告: DTW计算较慢，交叉验证可能需要较长时间")

    # 使用StratifiedKFold确保每折中类别分布一致
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_accuracies = []
    cv_f1_scores = []
    fold_idx = 1

    for train_idx, val_idx in skf.split(sequences, labels):
        print(f"\n  折 {fold_idx}/{N_FOLDS}:")

        # 划分数据
        X_cv_train = [sequences[i] for i in train_idx]
        y_cv_train = [labels[i] for i in train_idx]
        X_cv_val = [sequences[i] for i in val_idx]
        y_cv_val = [labels[i] for i in val_idx]

        # 特征缩放
        if SCALE_FEATURES:
            scaler_cv = StandardScaler()
            all_frames_cv = np.vstack(X_cv_train)
            scaler_cv.fit(all_frames_cv)
            X_cv_train = [scaler_cv.transform(x) for x in X_cv_train]
            X_cv_val = [scaler_cv.transform(x) for x in X_cv_val]

        # 训练和预测
        clf_cv = DTW_KNN_Classifier(n_neighbors=K_NEIGHBORS)
        clf_cv.fit(X_cv_train, y_cv_train)
        y_cv_pred = clf_cv.predict(X_cv_val)

        # 计算指标
        fold_acc = accuracy_score(y_cv_val, y_cv_pred)
        fold_f1 = f1_score(y_cv_val, y_cv_pred, average="macro")

        cv_accuracies.append(fold_acc)
        cv_f1_scores.append(fold_f1)

        print(f"    准确率: {fold_acc:.3f}")
        print(f"    宏平均F1: {fold_f1:.3f}")

        fold_idx += 1

    # 计算统计量
    mean_acc = np.mean(cv_accuracies)
    std_acc = np.std(cv_accuracies)
    mean_f1 = np.mean(cv_f1_scores)
    std_f1 = np.std(cv_f1_scores)

    print("\n" + "-" * 70)
    print("交叉验证结果汇总:")
    print("-" * 70)
    print(f"平均准确率: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"准确率范围: [{min(cv_accuracies):.3f}, {max(cv_accuracies):.3f}]")
    print(f"\n平均宏F1分数: {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"F1分数范围: [{min(cv_f1_scores):.3f}, {max(cv_f1_scores):.3f}]")
    print("-" * 70)

    # 保存交叉验证结果
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
    test_size=0.05,
    random_state=RANDOM_STATE,
    stratify=labels,
)
print(f"\n训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ===============================
# 特征缩放
# ===============================
if SCALE_FEATURES:
    scaler = StandardScaler()
    all_train_frames = np.vstack(X_train)
    scaler.fit(all_train_frames)
    X_train = [scaler.transform(x) for x in X_train]
    X_test = [scaler.transform(x) for x in X_test]
    print("特征标准化完成")

# ===============================
# 训练DTW-KNN模型
# ===============================
print("\n正在训练 DTW-KNN 模型...")
classifier = DTW_KNN_Classifier(n_neighbors=K_NEIGHBORS)
classifier.fit(X_train, y_train)
print(f"✅ DTW-KNN 模型训练完成（存储了 {len(X_train)} 个训练样本）")

# ===============================
# 预测与置信度分析
# ===============================
print(f"\n使用预测策略: {PREDICTION_STRATEGY}")
print("开始预测（计算DTW距离可能需要一些时间）...")

y_pred = []
confidence_results = []
unique_labels = sorted(set(y_train))

for i, seq in enumerate(X_test):
    if (i + 1) % 10 == 0:
        print(f"  进度: {i+1}/{len(X_test)}")

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

print("✅ 预测完成！")

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

cm = confusion_matrix(y_test, y_pred_filtered, labels=labels_sorted)
acc = accuracy_score(y_test, y_pred_filtered)

print("\n" + "=" * 70)
print("=== 基本分类结果 ===")
print("=" * 70)
print(f"准确率: {acc:.3f}")
print(f"\n混淆矩阵:\n{cm}")

# 获取详细分类报告（包含F1分数）
report_dict = classification_report(
    y_test, y_pred_filtered, target_names=labels_sorted, output_dict=True
)
print(
    f"\n详细报告:\n{classification_report(y_test, y_pred_filtered, target_names=labels_sorted)}"
)

# 提取各类别F1分数
print("\n" + "=" * 70)
print("=== 各类别F1分数 ===")
print("=" * 70)
print(f"{'类别':<15} {'F1-Score':<12} {'Precision':<12} {'Recall':<12} {'Support'}")
print("-" * 70)
for label in labels_sorted:
    f1 = report_dict[label]["f1-score"]
    precision = report_dict[label]["precision"]
    recall = report_dict[label]["recall"]
    support = report_dict[label]["support"]
    print(f"{label:<15} {f1:<12.3f} {precision:<12.3f} {recall:<12.3f} {support}")

# 宏平均和加权平均
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

print(f"\n整体置信度统计:")
print(f"  平均置信度: {confidence_df['confidence_score'].mean():.3f}")
print(f"  置信度中位数: {confidence_df['confidence_score'].median():.3f}")
print(f"  置信度标准差: {confidence_df['confidence_score'].std():.3f}")

print(f"\nDTW距离统计:")
print(f"  平均DTW距离: {confidence_df['avg_distance'].mean():.2f}")
print(
    f"  最小DTW距离范围: {confidence_df['min_distance'].min():.2f} ~ {confidence_df['min_distance'].max():.2f}"
)

correct_conf = confidence_df[confidence_df["is_correct"]]["confidence_score"].mean()
incorrect_conf = confidence_df[~confidence_df["is_correct"]]["confidence_score"].mean()
print(f"\n正确vs错误预测的置信度对比:")
print(f"  正确预测平均置信度: {correct_conf:.3f}")
print(f"  错误预测平均置信度: {incorrect_conf:.3f}")
print(f"  差异: {correct_conf - incorrect_conf:.3f}")

correct_dist = confidence_df[confidence_df["is_correct"]]["avg_distance"].mean()
incorrect_dist = confidence_df[~confidence_df["is_correct"]]["avg_distance"].mean()
print(f"\n正确vs错误预测的DTW距离对比:")
print(f"  正确预测平均距离: {correct_dist:.2f}")
print(f"  错误预测平均距离: {incorrect_dist:.2f}")

# 保存分析结果
output_file = "confidence_analysis_dtw_knn.csv"
confidence_df.to_csv(output_file, index=False)
print(f"\n💾 完整分析结果已保存至: {output_file}")

print("\n" + "=" * 70)
print("=== 💡 DTW-KNN 特点说明 ===")
print("=" * 70)
print(
    f"""
优点:
  1. 无需训练，直接基于样本对比
  2. 对时间扭曲不敏感，适合不同速度的动作
  3. 可解释性强（可查看最近邻样本）
  4. 对小样本友好

缺点:
  1. 预测速度较慢（需要计算所有训练样本的DTW距离）
  2. 内存占用大（需要存储所有训练样本）
  3. 对特征缩放敏感

调优建议:
  1. 调整K值（当前: {K_NEIGHBORS}）
  2. 设置DTW窗口大小限制（加速计算）
  3. 尝试不同特征组合
  4. 考虑使用FastDTW加速

模型性能总结:
  - 测试集准确率: {acc:.3f}
  - 测试集宏平均F1: {macro_f1:.3f}"""
)

if PERFORM_CV and len(sequences) >= N_FOLDS:
    print(f"  - 交叉验证准确率: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"  - 交叉验证宏F1: {mean_f1:.3f} ± {std_f1:.3f}")

print("=" * 70)
