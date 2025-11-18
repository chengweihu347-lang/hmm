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
from hmmlearn.hmm import GMMHMM
import warnings

warnings.filterwarnings("ignore")

# ===============================
# 参数设置
# ===============================
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "demobb")
LABEL_FILE = os.path.join(os.path.dirname(__file__), "labels.csv")

N_COMPONENTS = 3
N_MIXTURES = 3
MAX_ITER = 100
COV_TYPE = "diag"
RANDOM_STATE = 42
SCALE_FEATURES = True
NAN_STRATEGY = "interpolate"
MIN_VALID_FRAMES = 10

# 置信度阈值设置
CONFIDENCE_THRESHOLD_LOW = 0.3
CONFIDENCE_THRESHOLD_HIGH = 0.7

# 预测策略选择: 'standard', 'confidence_weighted', 'margin_based'
PREDICTION_STRATEGY = "standard"

# 序列长度归一化：True=使用平均似然，False=使用总似然
USE_NORMALIZED_LIKELIHOOD = True  # 推荐True，解决长度不等问题

# 交叉验证设置
USE_CROSS_VALIDATION = True  # 是否进行交叉验证
N_FOLDS = 5  # 交叉验证折数

# ===============================
# !!! 新增：特征权重设置 !!!
# (0:shoulder, 1:elbow, 2:hip, 3:knee, 4:ankle)
# ===============================
ANKLE_FEATURE_INDEX = 4
ANKLE_WEIGHT = 1  # 权重值 (1.0 = 不变, 2.0 = 2倍权重) 高/低效果都差


np.random.seed(RANDOM_STATE)


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

    return seq


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
    if len(seq) < MIN_VALID_FRAMES:
        return False
    if np.isnan(seq).all():
        return False
    return True


# ===============================
# !!! 新增：特征加权辅助函数 !!!
# ===============================
def apply_feature_weights(seq_list, feature_index, weight):
    """
    在标准化后对特定特征应用权重
    """
    if weight == 1.0:
        return seq_list

    weighted_seq_list = []
    for seq in seq_list:
        new_seq = seq.copy()
        # 对指定列（特征）乘以权重
        new_seq[:, feature_index] *= weight
        weighted_seq_list.append(new_seq)
    return weighted_seq_list


# ===============================
# 改进的似然计算（处理序列长度不等）
# ===============================
def compute_normalized_likelihood(model, seq, use_normalized=True):
    """
    计算归一化的似然值，解决序列长度不等的问题
    """
    raw_log_likelihood = model.score(seq)

    if use_normalized:
        # 使用平均似然（除以序列长度）
        normalized_likelihood = raw_log_likelihood / len(seq)
        return normalized_likelihood
    else:
        # 使用原始总似然
        return raw_log_likelihood


# ===============================
# 置信度计算函数
# ===============================
def calculate_confidence_metrics(log_likelihoods):
    """
    计算多种置信度指标
    """
    sorted_scores = sorted(log_likelihoods.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = sorted_scores[0]
    second_best_score = sorted_scores[1][1] if len(sorted_scores) > 1 else -np.inf

    # 1. 标准化概率（Softmax）
    scores = np.array(list(log_likelihoods.values()))
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / np.sum(exp_scores)
    max_probability = np.max(probabilities)

    # 2. Margin
    margin = best_score - second_best_score

    # 3. Entropy
    probabilities_safe = probabilities + 1e-10
    entropy = -np.sum(probabilities * np.log(probabilities_safe))
    max_entropy = np.log(len(probabilities))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # 4. 综合置信度分数
    confidence_score = max_probability * (1 - normalized_entropy)

    return {
        "predicted_label": best_label,
        "max_probability": max_probability,
        "margin": margin,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "confidence_score": confidence_score,
        "all_probabilities": dict(zip(log_likelihoods.keys(), probabilities)),
        "raw_scores": log_likelihoods,
    }


def predict_with_strategy(seq, models, strategy="standard"):
    """
    根据不同策略进行预测（使用归一化似然）
    """
    # 使用归一化似然计算
    logL = {
        label: compute_normalized_likelihood(
            models[label], seq, USE_NORMALIZED_LIKELIHOOD
        )
        for label in models
    }
    confidence_metrics = calculate_confidence_metrics(logL)

    if strategy == "standard":
        pred_label = confidence_metrics["predicted_label"]
    elif strategy == "confidence_weighted":
        if confidence_metrics["confidence_score"] < CONFIDENCE_THRESHOLD_LOW:
            pred_label = "UNCERTAIN"
        else:
            pred_label = confidence_metrics["predicted_label"]
    elif strategy == "margin_based":
        if confidence_metrics["margin"] < 5.0:
            pred_label = "UNCERTAIN"
        else:
            pred_label = confidence_metrics["predicted_label"]
    else:
        pred_label = confidence_metrics["predicted_label"]

    return pred_label, confidence_metrics


# ===============================
# 交叉验证函数
# ===============================
def perform_cross_validation(sequences, labels, n_folds=5):
    """
    执行交叉验证，计算平均准确率和标准差
    """
    print(f"\n{'='*70}")
    print(f"执行 {n_folds} 折交叉验证...")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_accuracies = []
    cv_f1_scores = []
    fold_num = 1

    for train_idx, val_idx in skf.split(sequences, labels):
        print(f"\n第 {fold_num}/{n_folds} 折:")

        # 划分数据
        X_train_fold = [sequences[i] for i in train_idx]
        y_train_fold = [labels[i] for i in train_idx]
        X_val_fold = [sequences[i] for i in val_idx]
        y_val_fold = [labels[i] for i in val_idx]

        # 特征标准化
        if SCALE_FEATURES:
            scaler = StandardScaler()
            all_train_frames = np.vstack(X_train_fold)
            scaler.fit(all_train_frames)
            X_train_fold = [scaler.transform(x) for x in X_train_fold]
            X_val_fold = [scaler.transform(x) for x in X_val_fold]

        # !!! 修改：在CV中应用特征权重 !!!
        if ANKLE_WEIGHT != 1.0:
            print(f"  (CV Fold {fold_num}: 正在应用 ankle 权重 {ANKLE_WEIGHT}x)")
            X_train_fold = apply_feature_weights(
                X_train_fold, ANKLE_FEATURE_INDEX, ANKLE_WEIGHT
            )
            X_val_fold = apply_feature_weights(
                X_val_fold, ANKLE_FEATURE_INDEX, ANKLE_WEIGHT
            )

        # 训练模型
        models_fold = {}
        unique_labels = sorted(set(y_train_fold))

        for label in unique_labels:
            label_seqs = [
                X_train_fold[i]
                for i in range(len(X_train_fold))
                if y_train_fold[i] == label
            ]
            X_concat = np.vstack(label_seqs)
            lengths = [len(x) for x in label_seqs]

            model = GMMHMM(
                n_components=N_COMPONENTS,
                n_mix=N_MIXTURES,
                covariance_type=COV_TYPE,
                n_iter=MAX_ITER,
                random_state=RANDOM_STATE,
                verbose=False,
            )
            model.fit(X_concat, lengths)
            models_fold[label] = model

        # 预测
        y_pred_fold = []
        for seq in X_val_fold:
            pred_label, _ = predict_with_strategy(seq, models_fold, "standard")
            y_pred_fold.append(pred_label)

        # 计算指标
        fold_acc = accuracy_score(y_val_fold, y_pred_fold)
        fold_f1 = f1_score(y_val_fold, y_pred_fold, average="macro", zero_division=0)

        cv_accuracies.append(fold_acc)
        cv_f1_scores.append(fold_f1)

        print(f"  准确率: {fold_acc:.4f}, F1分数: {fold_f1:.4f}")
        fold_num += 1

    return cv_accuracies, cv_f1_scores


# ===============================
# 加载数据
# ===============================
if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError(f"标签文件未找到: {LABEL_FILE}")

labels_df = pd.read_csv(LABEL_FILE)
print(f"Loaded labels.csv: {len(labels_df)} entries")

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

# 显示序列长度统计
seq_lengths = [len(seq) for seq in sequences]
print(f"\n序列长度统计:")
print(f"  最短: {min(seq_lengths)} 帧")
print(f"  最长: {max(seq_lengths)} 帧")
print(f"  平均: {np.mean(seq_lengths):.1f} 帧")
print(f"  标准差: {np.std(seq_lengths):.1f} 帧")
print(f"\n使用归一化似然: {'是' if USE_NORMALIZED_LIKELIHOOD else '否'}")

# ===============================
# 交叉验证（可选）
# ===============================
if USE_CROSS_VALIDATION and len(sequences) >= 10:
    cv_accuracies, cv_f1_scores = perform_cross_validation(sequences, labels, N_FOLDS)

    print(f"\n{'='*70}")
    print("=== 交叉验证结果汇总 ===")
    print(f"{'='*70}")
    print(f"\n准确率:")
    print(f"  各折: {[f'{acc:.4f}' for acc in cv_accuracies]}")
    print(f"  平均: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
    print(f"  最小: {np.min(cv_accuracies):.4f}")
    print(f"  最大: {np.max(cv_accuracies):.4f}")

    print(f"\nF1分数:")
    print(f"  各折: {[f'{f1:.4f}' for f1 in cv_f1_scores]}")
    print(f"  平均: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")
    print(f"  最小: {np.min(cv_f1_scores):.4f}")
    print(f"  最大: {np.max(cv_f1_scores):.4f}")

    # 保存交叉验证结果
    cv_results = pd.DataFrame(
        {
            "fold": range(1, N_FOLDS + 1),
            "accuracy": cv_accuracies,
            "f1_score": cv_f1_scores,
        }
    )
    cv_results.to_csv("cross_validation_results.csv", index=False)
    print(f"\n💾 交叉验证结果已保存至: cross_validation_results.csv")

# ===============================
# 划分数据集
# ===============================
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    sequences,
    labels,
    video_ids,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=labels,
)
print(f"\n{'='*70}")
print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

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

# !!! 修改：应用特征权重到 训练集/测试集 !!!
if ANKLE_WEIGHT != 1.0:
    print(f"应用 ankle 权重 {ANKLE_WEIGHT}x 到 训练集/测试集...")
    X_train = apply_feature_weights(X_train, ANKLE_FEATURE_INDEX, ANKLE_WEIGHT)
    X_test = apply_feature_weights(X_test, ANKLE_FEATURE_INDEX, ANKLE_WEIGHT)

# ===============================
# 训练模型
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
# 预测与置信度分析
# ===============================
print(f"\n使用预测策略: {PREDICTION_STRATEGY}")
y_pred = []
confidence_results = []

for i, seq in enumerate(X_test):
    pred_label, conf_metrics = predict_with_strategy(seq, models, PREDICTION_STRATEGY)
    y_pred.append(pred_label)

    confidence_results.append(
        {
            "video_id": ids_test[i],
            "true_label": y_test[i],
            "predicted_label": pred_label,
            "sequence_length": len(seq),
            "confidence_score": conf_metrics["confidence_score"],
            "max_probability": conf_metrics["max_probability"],
            "margin": conf_metrics["margin"],
            "entropy": conf_metrics["normalized_entropy"],
            "is_correct": pred_label == y_test[i],
            "all_probs": conf_metrics["all_probabilities"],
        }
    )

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

# 计算各类别的F1分数
f1_per_class = f1_score(
    y_test, y_pred_filtered, labels=labels_sorted, average=None, zero_division=0
)
f1_macro = f1_score(y_test, y_pred_filtered, average="macro", zero_division=0)
f1_weighted = f1_score(y_test, y_pred_filtered, average="weighted", zero_division=0)

print("\n" + "=" * 70)
print("=== 基本分类结果 ===")
print("=" * 70)
print(f"准确率: {acc:.3f}")
print(f"Macro F1分数: {f1_macro:.3f}")
print(f"Weighted F1分数: {f1_weighted:.3f}")

print(f"\n各类别F1分数:")
for label, f1 in zip(labels_sorted, f1_per_class):
    print(f"  {label}: {f1:.3f}")

print(f"\n混淆矩阵:\n{cm}")
print(
    f"\n详细报告:\n{classification_report(y_test, y_pred_filtered, target_names=labels_sorted)}"
)

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
print(f"  最低置信度: {confidence_df['confidence_score'].min():.3f}")
print(f"  最高置信度: {confidence_df['confidence_score'].max():.3f}")

# 按置信度区间统计
print(f"\n按置信度区间分布:")
conf_bins = [0, 0.3, 0.5, 0.7, 1.0]
conf_labels = ["很低(0-0.3)", "低(0.3-0.5)", "中(0.5-0.7)", "高(0.7-1.0)"]
confidence_df["conf_bin"] = pd.cut(
    confidence_df["confidence_score"], bins=conf_bins, labels=conf_labels
)
print(confidence_df["conf_bin"].value_counts().sort_index())

# 正确vs错误预测的置信度对比
print(f"\n正确vs错误预测的置信度对比:")
correct_conf = confidence_df[confidence_df["is_correct"]]["confidence_score"].mean()
incorrect_conf = confidence_df[~confidence_df["is_correct"]]["confidence_score"].mean()
print(f"  正确预测平均置信度: {correct_conf:.3f}")
print(f"  错误预测平均置信度: {incorrect_conf:.3f}")
print(f"  差异: {correct_conf - incorrect_conf:.3f}")

# ===============================
# 各类别表现分析（增强版）
# ===============================
print("\n" + "=" * 70)
print("=== 各类别详细表现分析 ===")
print("=" * 70)

class_performance = []

for idx, label in enumerate(labels_sorted):
    label_data = confidence_df[confidence_df["true_label"] == label]

    print(f"\n📊 类别: {label}")
    print(f"{'='*60}")

    # 基本统计
    total = len(label_data)
    correct = label_data["is_correct"].sum()
    accuracy = correct / total if total > 0 else 0
    f1 = f1_per_class[idx]

    print(f"样本数: {total}")
    print(f"准确率: {accuracy:.3f} ({correct}/{total})")
    print(f"F1分数: {f1:.3f}")

    # 序列长度统计
    avg_length = label_data["sequence_length"].mean()
    print(f"平均序列长度: {avg_length:.1f} 帧")

    # 置信度统计
    avg_conf = label_data["confidence_score"].mean()
    print(f"平均置信度: {avg_conf:.3f}")

    # 正确和错误样本的置信度
    if correct > 0:
        correct_avg = label_data[label_data["is_correct"]]["confidence_score"].mean()
        print(f"  ├─ 正确预测: {correct_avg:.3f}")
    if correct < total:
        incorrect_avg = label_data[~label_data["is_correct"]]["confidence_score"].mean()
        print(f"  └─ 错误预测: {incorrect_avg:.3f}")

    # 混淆情况
    if correct < total:
        confused_with = label_data[~label_data["is_correct"]][
            "predicted_label"
        ].value_counts()
        print(f"主要混淆为:")
        for conf_label, count in confused_with.items():
            print(f"  → {conf_label}: {count} 次")

    # 保存类别性能数据
    class_performance.append(
        {
            "class": label,
            "samples": total,
            "accuracy": accuracy,
            "f1_score": f1,
            "avg_confidence": avg_conf,
            "avg_length": avg_length,
        }
    )

# 保存类别性能汇总
class_perf_df = pd.DataFrame(class_performance)
class_perf_df.to_csv("per_class_performance.csv", index=False)
print(f"\n💾 各类别性能汇总已保存至: per_class_performance.csv")

# ===============================
# 低置信度样本标注
# ===============================
print("\n" + "=" * 70)
print("=== 低置信度样本（需人工复核）===")
print("=" * 70)

low_conf_samples = confidence_df[
    confidence_df["confidence_score"] < CONFIDENCE_THRESHOLD_LOW
].sort_values("confidence_score")

print(
    f"\n发现 {len(low_conf_samples)} 个低置信度样本 (阈值 < {CONFIDENCE_THRESHOLD_LOW}):"
)
print(
    f"\n{'视频ID':<30} {'真实':<10} {'预测':<10} {'长度':<8} {'置信度':<10} {'正确?'}"
)
print("-" * 80)

for _, row in low_conf_samples.head(20).iterrows():
    check = "✓" if row["is_correct"] else "✗"
    print(
        f"{row['video_id']:<30} {row['true_label']:<10} {row['predicted_label']:<10} "
        f"{row['sequence_length']:<8} {row['confidence_score']:<10.3f} {check}"
    )

low_conf_file = "low_confidence_samples.csv"
low_conf_samples.to_csv(low_conf_file, index=False)
print(f"\n💾 低置信度样本已保存至: {low_conf_file}")

# ===============================
# 高置信度错误样本
# ===============================
print("\n" + "=" * 70)
print("=== 高置信度但预测错误的样本（重点！）===")
print("=" * 70)

high_conf_errors = confidence_df[
    (~confidence_df["is_correct"])
    & (confidence_df["confidence_score"] > CONFIDENCE_THRESHOLD_HIGH)
].sort_values("confidence_score", ascending=False)

print(
    f"\n发现 {len(high_conf_errors)} 个高置信度错误样本 (置信度 > {CONFIDENCE_THRESHOLD_HIGH}):"
)
print("这些样本可能存在标签错误或特征提取问题！\n")

if len(high_conf_errors) > 0:
    print(f"{'视频ID':<30} {'真实':<10} {'预测':<10} {'长度':<8} {'置信度':<10}")
    print("-" * 80)
    for _, row in high_conf_errors.head(10).iterrows():
        print(
            f"{row['video_id']:<30} {row['true_label']:<10} {row['predicted_label']:<10} "
            f"{row['sequence_length']:<8} {row['confidence_score']:<10.3f}"
        )

    high_conf_errors.to_csv("high_confidence_errors.csv", index=False)
    print(f"\n💾 高置信度错误样本已保存至: high_confidence_errors.csv")

# ===============================
# 综合改进建议
# ===============================
print("\n" + "=" * 70)
print("=== 💡 改进建议 ===")
print("=" * 70)

# 1. 找出表现最差的类别
worst_class_idx = np.argmin([perf["accuracy"] for perf in class_performance])
worst_class_info = class_performance[worst_class_idx]

print(f"\n1. 优先改进类别: {worst_class_info['class']}")
print(f"   准确率: {worst_class_info['accuracy']:.3f}")
print(f"   F1分数: {worst_class_info['f1_score']:.3f}")
print(f"   建议: 增加该类别训练样本，检查标注质量")

# 2. 数据质量问题
print(f"\n2. 数据质量检查:")
print(f"   - 复核 {len(high_conf_errors)} 个高置信度错误样本的标签")
print(f"   - 检查 {len(low_conf_samples)} 个低置信度样本的特征质量")

# 3. 模型调优方向
avg_margin = confidence_df["margin"].mean()
print(f"\n3. 模型调优建议:")
print(f"   - 当前平均Margin: {avg_margin:.2f}")
if avg_margin < 10:
    print(f"   - Margin较小，建议增加N_COMPONENTS或N_MIXTURES")
if incorrect_conf > 0.6:
    print(f"   - 错误预测置信度较高，可能存在过拟合")
    print(f"   - 建议: 减小模型复杂度或增加训练数据")

# 4. 序列长度归一化效果
if USE_NORMALIZED_LIKELIHOOD:
    print(f"\n4. 序列长度归一化:")
    print(f"   - 已启用归一化似然（除以序列长度）")
    # 分析长短序列的预测表现
    median_length = confidence_df["sequence_length"].median()
    short_seqs = confidence_df[confidence_df["sequence_length"] < median_length]
    long_seqs = confidence_df[confidence_df["sequence_length"] >= median_length]

    if len(short_seqs) > 0 and len(long_seqs) > 0:
        short_acc = short_seqs["is_correct"].mean()
        long_acc = long_seqs["is_correct"].mean()
        print(f"   - 短序列(<{median_length:.0f}帧)准确率: {short_acc:.3f}")
        print(f"   - 长序列(≥{median_length:.0f}帧)准确率: {long_acc:.3f}")
        print(f"   - 差异: {abs(short_acc - long_acc):.3f}")

# 保存完整分析结果
full_analysis_file = "confidence_analysis_full.csv"
confidence_df.to_csv(full_analysis_file, index=False)
print(f"\n💾 完整分析结果已保存至: {full_analysis_file}")

print("\n" + "=" * 70)
print("分析完成！")
print("=" * 70)
