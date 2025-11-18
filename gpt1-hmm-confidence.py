import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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
CONFIDENCE_THRESHOLD_LOW = 0.3  # 低置信度阈值
CONFIDENCE_THRESHOLD_HIGH = 0.7  # 高置信度阈值

# 预测策略选择: 'standard', 'confidence_weighted', 'margin_based'
PREDICTION_STRATEGY = "standard"

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
# 置信度计算函数
# ===============================
def calculate_confidence_metrics(log_likelihoods):
    """
    计算多种置信度指标w

    Args:
        log_likelihoods: dict, {label: log_likelihood}

    Returns:
        dict: 包含多种置信度指标
    """
    sorted_scores = sorted(log_likelihoods.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = sorted_scores[0]
    second_best_score = sorted_scores[1][1] if len(sorted_scores) > 1 else -np.inf

    # 1. 标准化概率（Softmax）
    scores = np.array(list(log_likelihoods.values()))
    # 为了数值稳定性，减去最大值
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / np.sum(exp_scores)
    max_probability = np.max(probabilities)

    # 2. Margin (最高分与第二高分的差距)
    margin = best_score - second_best_score

    # 3. Entropy (不确定性度量)
    # 避免log(0)
    probabilities_safe = probabilities + 1e-10
    entropy = -np.sum(probabilities * np.log(probabilities_safe))
    # 归一化entropy到[0,1]
    max_entropy = np.log(len(probabilities))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # 4. 综合置信度分数 (0-1之间)
    # 结合probability和margin
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
    根据不同策略进行预测

    Args:
        seq: 输入序列
        models: 训练好的HMM模型字典
        strategy: 预测策略

    Returns:
        predicted_label, confidence_metrics
    """
    logL = {label: models[label].score(seq) for label in models}
    confidence_metrics = calculate_confidence_metrics(logL)

    if strategy == "standard":
        # 标准策略：选择最高分
        pred_label = confidence_metrics["predicted_label"]

    elif strategy == "confidence_weighted":
        # 置信度加权策略：低置信度时返回"uncertain"
        if confidence_metrics["confidence_score"] < CONFIDENCE_THRESHOLD_LOW:
            pred_label = "UNCERTAIN"
        else:
            pred_label = confidence_metrics["predicted_label"]

    elif strategy == "margin_based":
        # 基于Margin的策略：margin太小时认为不确定
        if confidence_metrics["margin"] < 5.0:  # 可调整阈值
            pred_label = "UNCERTAIN"
        else:
            pred_label = confidence_metrics["predicted_label"]

    else:
        pred_label = confidence_metrics["predicted_label"]

    return pred_label, confidence_metrics


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

    # 保存详细的置信度信息
    confidence_results.append(
        {
            "video_id": ids_test[i],
            "true_label": y_test[i],
            "predicted_label": pred_label,
            "confidence_score": conf_metrics["confidence_score"],
            "max_probability": conf_metrics["max_probability"],
            "margin": conf_metrics["margin"],
            "entropy": conf_metrics["normalized_entropy"],
            "is_correct": pred_label == y_test[i],
            "all_probs": conf_metrics["all_probabilities"],
        }
    )

# 创建置信度DataFrame
confidence_df = pd.DataFrame(confidence_results)

# ===============================
# 基本评估结果
# ===============================
labels_sorted = sorted(unique_labels)

# 处理UNCERTAIN标签（如果有）
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
# 各类别表现分析
# ===============================
print("\n" + "=" * 70)
print("=== 各类别详细表现分析 ===")
print("=" * 70)

for label in labels_sorted:
    label_data = confidence_df[confidence_df["true_label"] == label]

    print(f"\n📊 类别: {label}")
    print(f"{'='*60}")

    # 基本统计
    total = len(label_data)
    correct = label_data["is_correct"].sum()
    accuracy = correct / total if total > 0 else 0

    print(f"样本数: {total}")
    print(f"准确率: {accuracy:.3f} ({correct}/{total})")

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
print(f"\n{'视频ID':<30} {'真实':<10} {'预测':<10} {'置信度':<10} {'正确?'}")
print("-" * 70)

for _, row in low_conf_samples.head(20).iterrows():  # 显示前20个
    check = "✓" if row["is_correct"] else "✗"
    print(
        f"{row['video_id']:<30} {row['true_label']:<10} {row['predicted_label']:<10} {row['confidence_score']:<10.3f} {check}"
    )

# 保存低置信度样本到文件
low_conf_file = "low_confidence_samples.csv"
low_conf_samples.to_csv(low_conf_file, index=False)
print(f"\n💾 低置信度样本已保存至: {low_conf_file}")

# ===============================
# 高置信度错误样本（重点关注）
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
    print(f"{'视频ID':<30} {'真实':<10} {'预测':<10} {'置信度':<10}")
    print("-" * 70)
    for _, row in high_conf_errors.head(10).iterrows():
        print(
            f"{row['video_id']:<30} {row['true_label']:<10} {row['predicted_label']:<10} {row['confidence_score']:<10.3f}"
        )

    high_conf_errors.to_csv("high_confidence_errors.csv", index=False)
    print(f"\n💾 高置信度错误样本已保存至: high_confidence_errors.csv")

# ===============================
# 边界样本分析
# ===============================
print("\n" + "=" * 70)
print("=== 边界样本分析（Margin小的样本）===")
print("=" * 70)

low_margin_samples = confidence_df.sort_values("margin").head(15)

print("\nMargin最小的15个样本（最难区分）:")
print(f"{'视频ID':<30} {'真实':<10} {'预测':<10} {'Margin':<10} {'正确?'}")
print("-" * 70)

for _, row in low_margin_samples.iterrows():
    check = "✓" if row["is_correct"] else "✗"
    print(
        f"{row['video_id']:<30} {row['true_label']:<10} {row['predicted_label']:<10} {row['margin']:<10.2f} {check}"
    )

# ===============================
# 综合改进建议
# ===============================
print("\n" + "=" * 70)
print("=== 💡 改进建议 ===")
print("=" * 70)

# 1. 找出表现最差的类别
worst_class = (
    confidence_df.groupby("true_label")
    .apply(lambda x: x["is_correct"].sum() / len(x))
    .sort_values()
    .index[0]
)

worst_acc = (
    confidence_df.groupby("true_label")
    .apply(lambda x: x["is_correct"].sum() / len(x))
    .sort_values()
    .values[0]
)

print(f"\n1. 优先改进类别: {worst_class} (准确率: {worst_acc:.3f})")
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

# 保存完整分析结果
full_analysis_file = "confidence_analysis_full.csv"
confidence_df.to_csv(full_analysis_file, index=False)
print(f"\n💾 完整分析结果已保存至: {full_analysis_file}")

print("\n" + "=" * 70)
