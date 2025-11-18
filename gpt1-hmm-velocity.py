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
from hmmlearn.hmm import GMMHMM
import warnings

warnings.filterwarnings("ignore")

# ===============================
# 参数设置
# ===============================
# 注意: 为了在Canvas环境中更稳定运行，通常推荐使用简单的相对路径或直接的文件名。
# 这里保留用户提供的os.path.join(os.path.dirname(__file__)...)结构。
DATA_FOLDER = "demobb"  # 简化路径
LABEL_FILE = "labels.csv"  # 简化路径

N_COMPONENTS = 3
N_MIXTURES = 3
MAX_ITER = 100
COV_TYPE = "diag"
RANDOM_STATE = 42
SCALE_FEATURES = True
NAN_STRATEGY = "interpolate"
MIN_VALID_FRAMES = 10

# ===============================
# 特征配置
# ===============================
USE_ANGLE = True  # 是否使用角度
USE_VELOCITY = False  # 是否使用角速度
USE_ACCELERATION = False  # 是否使用角加速度

# ===============================
# 评估与策略
# ===============================
# 序列长度归一化：True=使用平均似然，False=使用总似然
USE_NORMALIZED_LIKELIHOOD = True  # 推荐True，解决长度不等问题

# 交叉验证设置
USE_CROSS_VALIDATION = True  # 是否进行交叉验证
N_FOLDS = 5  # 交叉验证折数

# 置信度阈值设置
CONFIDENCE_THRESHOLD_LOW = 0.3
CONFIDENCE_THRESHOLD_HIGH = 0.7

# 预测策略选择: 'standard', 'confidence_weighted', 'margin_based'
PREDICTION_STRATEGY = "standard"

np.random.seed(RANDOM_STATE)


# ===============================
# 特征提取函数
# ===============================
def compute_derivatives(angles):
    """
    计算角度的一阶导数(角速度)和二阶导数(角加速度)
    """
    # 使用numpy的gradient计算导数，更稳定
    velocity = np.gradient(angles, axis=0)
    acceleration = np.gradient(velocity, axis=0)

    return velocity, acceleration


def extract_features(seq):
    """
    根据配置提取特征：角度、角速度、角加速度
    """
    features_list = []

    # 1. 原始角度
    if USE_ANGLE:
        features_list.append(seq)

    # 2. 角速度（一阶导数）
    if USE_VELOCITY:
        velocity, _ = compute_derivatives(seq)
        features_list.append(velocity)

    # 3. 角加速度（二阶导数）
    if USE_ACCELERATION:
        _, acceleration = compute_derivatives(seq)
        features_list.append(acceleration)

    # 拼接所有特征
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

    # 提取特征（包含角加速度）
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
    if len(seq) < MIN_VALID_FRAMES:
        return False
    if np.isnan(seq).all():
        return False
    # HMM模型对inf/nan值敏感，确保序列不含这些值
    if not np.isfinite(seq).all():
        return False
    return True


# ===============================
# 改进的似然计算（处理序列长度不等）
# ===============================
def compute_normalized_likelihood(model, seq, use_normalized=True):
    """
    计算归一化的似然值，解决序列长度不等的问题
    """
    try:
        raw_log_likelihood = model.score(seq)
    except Exception as e:
        # print(f"  [Warning] model.score() failed: {e}. Returning -inf.")
        return -np.inf

    if use_normalized:
        # 使用平均似然（除以序列长度）
        if len(seq) == 0:
            return -np.inf
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
    # 过滤掉 -inf 的似然值，但保留对应的标签
    filtered_scores = {
        k: v for k, v in log_likelihoods.items() if v > -np.inf and np.isfinite(v)
    }

    if not filtered_scores:
        # 所有模型都失败的情况
        all_labels = list(log_likelihoods.keys())
        default_prob = 1.0 / len(all_labels) if all_labels else 0
        return {
            "predicted_label": "UNKNOWN",
            "max_probability": default_prob,
            "margin": 0.0,
            "entropy": np.log(len(all_labels)) if all_labels else 0,
            "normalized_entropy": 1.0,
            "confidence_score": 0.0,
            "all_probabilities": {label: default_prob for label in all_labels},
            "raw_scores": log_likelihoods,
        }

    sorted_scores = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = sorted_scores[0]
    second_best_score = (
        sorted_scores[1][1] if len(sorted_scores) > 1 else best_score - 100
    )  # 如果只有一个有效模型，给一个大差距

    # 1. 标准化概率（Softmax）
    scores = np.array(list(filtered_scores.values()))

    # 防御-inf，但由于前面过滤了，这里主要处理数值稳定性
    if not scores.size:
        # 应该不会发生，但作为防御性编程
        best_label = "UNKNOWN"
        scores = np.array([0])
        max_probability = 0.0
    else:
        # 对数-和-指数技巧 (LogSumExp Trick) 确保数值稳定
        max_score = np.max(scores)
        with np.errstate(over="ignore", under="ignore"):
            exp_scores = np.exp(scores - max_score)

        sum_exp_scores = np.sum(exp_scores)

        if sum_exp_scores == 0 or not np.isfinite(sum_exp_scores):
            probabilities = np.zeros_like(scores)
            probabilities[np.argmax(scores)] = 1.0
        else:
            probabilities = exp_scores / sum_exp_scores

        max_probability = np.max(probabilities)

    # 重新映射回所有标签
    all_probabilities = {label: 0.0 for label in log_likelihoods.keys()}
    for label, prob in zip(filtered_scores.keys(), probabilities):
        all_probabilities[label] = prob

    # 2. Margin
    margin = best_score - second_best_score

    # 3. Entropy
    probs = np.array(list(all_probabilities.values()))  # 使用所有标签的概率
    probabilities_safe = probs + 1e-10
    entropy = -np.sum(probs * np.log(probabilities_safe))
    max_entropy = np.log(len(all_probabilities))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # 4. 综合置信度分数
    # 置信度高 = 最大概率高 * 熵低 (低不确定性)
    confidence_score = max_probability * (1 - normalized_entropy)

    return {
        "predicted_label": best_label,
        "max_probability": max_probability,
        "margin": margin,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "confidence_score": confidence_score,
        "all_probabilities": all_probabilities,
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
        # Margin阈值需要根据似然分数范围调整，5.0是一个经验值
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

    # 确保 labels 是 numpy 数组
    labels_array = np.array(labels)

    for train_idx, val_idx in skf.split(sequences, labels_array):
        print(f"\n第 {fold_num}/{n_folds} 折:")

        # 划分数据
        X_train_fold = [sequences[i] for i in train_idx]
        y_train_fold = labels_array[train_idx]
        X_val_fold = [sequences[i] for i in val_idx]
        y_val_fold = labels_array[val_idx]

        # 特征标准化 (在fold内部fit，防止泄露)
        if SCALE_FEATURES:
            scaler = StandardScaler()
            all_train_frames = np.vstack(X_train_fold)
            scaler.fit(all_train_frames)
            X_train_fold = [scaler.transform(x) for x in X_train_fold]
            X_val_fold = [scaler.transform(x) for x in X_val_fold]

        # 训练模型
        models_fold = {}
        unique_labels = sorted(set(y_train_fold))

        for label in unique_labels:
            label_seqs = [
                X_train_fold[i]
                for i in range(len(X_train_fold))
                if y_train_fold[i] == label
            ]
            if not label_seqs:
                print(
                    f"  [Warning] CV Fold {fold_num}: 类别 {label} 没有训练样本，跳过。"
                )
                continue

            # 过滤掉不合法的序列，防止模型训练失败
            valid_seqs = [seq for seq in label_seqs if is_valid_sequence(seq)]
            if not valid_seqs:
                print(
                    f"  [Warning] CV Fold {fold_num}: 类别 {label} 有效序列太少，跳过。"
                )
                continue

            X_concat = np.vstack(valid_seqs)
            lengths = [len(x) for x in valid_seqs]

            # 强制要求HMM的特征维度和数据维度一致
            if X_concat.shape[1] != feature_dim:
                print(
                    f"  [Warning] CV Fold {fold_num}: 维度不匹配 {X_concat.shape[1]} != {feature_dim}，跳过。"
                )
                continue

            model = GMMHMM(
                n_components=N_COMPONENTS,
                n_mix=N_MIXTURES,
                covariance_type=COV_TYPE,
                n_iter=MAX_ITER,
                random_state=RANDOM_STATE,
                verbose=False,
            )
            try:
                model.fit(X_concat, lengths)
                models_fold[label] = model
            except Exception as e:
                print(f"  [Error] CV Fold {fold_num}: 训练 {label} 模型失败: {e}")

        # 预测
        y_pred_fold = []
        for seq in X_val_fold:
            # 确保模型被训练了
            if not models_fold:
                pred_label = y_val_fold[0]  # 回退到第一个标签
            else:
                pred_label, _ = predict_with_strategy(seq, models_fold, "standard")
            y_pred_fold.append(pred_label)

        # 移除 'UNKNOWN' 或 'N/A' 等无效预测，仅评估有效标签
        valid_labels_in_fold = list(models_fold.keys())

        # 筛选出在模型中存在的标签，否则classification_report会报错
        y_true_valid = [
            y_val_fold[i]
            for i, pred in enumerate(y_pred_fold)
            if pred in valid_labels_in_fold
        ]
        y_pred_valid = [pred for pred in y_pred_fold if pred in valid_labels_in_fold]

        if len(y_true_valid) == 0:
            fold_acc = 0.0
            fold_f1 = 0.0
        else:
            fold_acc = accuracy_score(y_true_valid, y_pred_valid)
            fold_f1 = f1_score(
                y_true_valid, y_pred_valid, average="macro", zero_division=0
            )

        cv_accuracies.append(fold_acc)
        cv_f1_scores.append(fold_f1)

        print(f"  有效样本准确率: {fold_acc:.4f}, Macro F1分数: {fold_f1:.4f}")
        fold_num += 1

    return cv_accuracies, cv_f1_scores


# ===============================
# 加载数据
# ===============================
print("=" * 70)
print("模型: HMM-GMM (Hidden Markov Model)")
print("参数:")
print(f"  - 隐状态数 (N_COMPONENTS): {N_COMPONENTS}")
print(f"  - 混合高斯数 (N_MIXTURES): {N_MIXTURES}")
print(f"  - 似然归一化 (USE_NORMALIZED_LIKELIHOOD): {USE_NORMALIZED_LIKELIHOOD}")
print("-" * 70)
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
            # print(f"[错误] 处理文件失败: {row['video_id']}, 错误: {e}")
            skipped_count += 1
    else:
        skipped_count += 1

print(f"\n已成功加载 {len(sequences)} 个样本")
print(f"跳过/失败: {skipped_count} 个样本")

label_counts = pd.Series(labels).value_counts()
print(f"\n各类别样本数:\n{label_counts}")

# 检查是否有足够的样本进行CV
if len(sequences) < N_FOLDS * 2 or len(label_counts) < 2:
    print(
        f"\n样本总数 ({len(sequences)}) 或类别数 ({len(label_counts)}) 过少，跳过交叉验证。"
    )
    USE_CROSS_VALIDATION = False

# ===============================
# 交叉验证（可选）
# ===============================
if USE_CROSS_VALIDATION:
    cv_accuracies, cv_f1_scores = perform_cross_validation(sequences, labels, N_FOLDS)

    print(f"\n{'='*70}")
    print("=== 交叉验证结果汇总 ===")
    print(f"{'='*70}")

    if cv_accuracies:
        print(f"\n准确率:")
        print(f"  各折: {[f'{acc:.4f}' for acc in cv_accuracies]}")
        print(
            f"  平均: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f} (标准差)"
        )
        print(f"  最小: {np.min(cv_accuracies):.4f}")
        print(f"  最大: {np.max(cv_accuracies):.4f}")

        print(f"\nMacro F1分数:")
        print(f"  各折: {[f'{f1:.4f}' for f1 in cv_f1_scores]}")
        print(
            f"  平均: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f} (标准差)"
        )
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
    else:
        print("⚠️ 交叉验证未能计算出有效结果。")

# ===============================
# 划分数据集 (用于最终测试)
# ===============================
if len(sequences) < 2:
    print("\n[Fatal Error] 样本数不足以划分训练集和测试集。退出。")
    exit()

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    sequences,
    labels,
    video_ids,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=labels,
)
print(f"\n{'='*70}")
print(f"最终测试集划分: 训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ===============================
# 特征缩放
# ===============================
if SCALE_FEATURES:
    scaler = StandardScaler()
    all_train_frames = np.vstack(X_train)
    scaler.fit(all_train_frames)
    X_train = [scaler.transform(x) for x in X_train]

    # 在测试集缩放时，同时记录序列长度
    X_test_scaled = []
    test_lengths = []
    for x in X_test:
        X_test_scaled.append(scaler.transform(x))
        test_lengths.append(len(x))
    X_test = X_test_scaled

    print("特征标准化完成 (基于训练集)")
else:
    test_lengths = [len(x) for x in X_test]

# ===============================
# 训练模型 (基于完整训练集)
# ===============================
models = {}
unique_labels = sorted(set(y_train))

for label in unique_labels:
    label_seqs = [X_train[i] for i in range(len(X_train)) if y_train[i] == label]

    if not label_seqs:
        continue

    valid_seqs = [seq for seq in label_seqs if is_valid_sequence(seq)]
    if not valid_seqs:
        continue

    X_concat = np.vstack(valid_seqs)
    lengths = [len(x) for x in valid_seqs]

    # 再次检查维度是否匹配
    if X_concat.shape[1] != feature_dim:
        print(
            f"  [Warning] 训练集维度不匹配 {X_concat.shape[1]} != {feature_dim}，跳过 {label} 模型训练。"
        )
        continue

    # print(f"\n正在训练 HMM 模型: {label} ({len(valid_seqs)} 条序列)...")
    model = GMMHMM(
        n_components=N_COMPONENTS,
        n_mix=N_MIXTURES,
        covariance_type=COV_TYPE,
        n_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    try:
        model.fit(X_concat, lengths)
        models[label] = model
        # print(f"✅ 完成 {label} 模型训练")
    except Exception as e:
        print(f"  [Error] 训练 {label} 模型失败: {e}")

if not models:
    print("\n[Fatal Error] 没有一个模型被成功训练。退出。")
    exit()

# ===============================
# 预测与置信度分析
# ===============================
print(f"\n使用预测策略: {PREDICTION_STRATEGY}")
y_pred = []
confidence_results = []

for i, seq in enumerate(X_test):
    # 如果序列不合法，则跳过
    if not is_valid_sequence(seq):
        pred_label = "UNKNOWN"
    else:
        pred_label, conf_metrics = predict_with_strategy(
            seq, models, PREDICTION_STRATEGY
        )

    # 仅在预测成功时添加置信度指标
    if pred_label != "UNKNOWN":
        confidence_results.append(
            {
                "video_id": ids_test[i],
                "true_label": y_test[i],
                "predicted_label": pred_label,
                "sequence_length": test_lengths[i],
                "confidence_score": conf_metrics["confidence_score"],
                "max_probability": conf_metrics["max_probability"],
                "margin": conf_metrics["margin"],
                "entropy": conf_metrics["normalized_entropy"],
                "is_correct": pred_label == y_test[i],
                "all_probs": conf_metrics["all_probabilities"],
            }
        )
    else:
        # 无法预测的样本也记录下来
        confidence_results.append(
            {
                "video_id": ids_test[i],
                "true_label": y_test[i],
                "predicted_label": "FAILED_TO_PREDICT",
                "sequence_length": test_lengths[i],
                "confidence_score": 0.0,
                "max_probability": 0.0,
                "margin": -np.inf,
                "entropy": 1.0,
                "is_correct": False,
                "all_probs": {},
            }
        )

confidence_df = pd.DataFrame(confidence_results)

# 过滤掉无法预测的样本
valid_predictions_df = confidence_df[
    confidence_df["predicted_label"] != "FAILED_TO_PREDICT"
].copy()

# ===============================
# 基本评估结果 (基于测试集)
# ===============================
labels_sorted = sorted(models.keys())  # 只评估训练过的标签

y_test_filtered = valid_predictions_df["true_label"].tolist()
y_pred_filtered = valid_predictions_df["predicted_label"].tolist()

if "UNCERTAIN" in y_pred_filtered:
    uncertain_count = y_pred_filtered.count("UNCERTAIN")
    print(f"\n发现 {uncertain_count} 个低置信度预测 (策略: {PREDICTION_STRATEGY})")
    # 评估时，将 'UNCERTAIN' 视为错误预测（如果真实标签不是 'UNCERTAIN'）
    y_pred_eval = [
        (
            p
            if p != "UNCERTAIN"
            else (
                y_test_filtered[i]
                if y_test_filtered[i] == "UNCERTAIN"
                else "UNCERTAIN_AS_ERROR"
            )
        )
        for i, p in enumerate(y_pred_filtered)
    ]
else:
    y_pred_eval = y_pred_filtered
    uncertain_count = 0


# 确保评估使用的标签集包含所有可能结果 (包括可能引入的 'UNCERTAIN_AS_ERROR')
all_eval_labels = sorted(list(set(y_test_filtered) | set(y_pred_eval)))
if "UNCERTAIN_AS_ERROR" in all_eval_labels:
    all_eval_labels.remove("UNCERTAIN_AS_ERROR")

cm = confusion_matrix(y_test_filtered, y_pred_eval, labels=all_eval_labels)
acc = accuracy_score(y_test_filtered, y_pred_eval)

report_dict = classification_report(
    y_test_filtered,
    y_pred_eval,
    target_names=all_eval_labels,
    output_dict=True,
    zero_division=0,
)
f1_macro = report_dict["macro avg"]["f1-score"]
f1_weighted = report_dict["weighted avg"]["f1-score"]

print("\n" + "=" * 70)
print("=== 基本分类结果 (测试集) ===")
print("=" * 70)
print(f"有效样本数: {len(valid_predictions_df)}")
print(f"准确率: {acc:.3f} (包含 'UNCERTAIN' 错误)")
print(f"Macro F1分数: {f1_macro:.3f}")

print(f"\n混淆矩阵 (行=真实, 列=预测):\n{cm}")
print(f"标签: {all_eval_labels}")
print(
    f"\n详细报告:\n{classification_report(y_test_filtered, y_pred_eval, target_names=all_eval_labels, zero_division=0)}"
)


# ===============================
# 置信度统计分析
# ===============================
print("\n" + "=" * 70)
print("=== 置信度分析 (测试集) ===")
print("=" * 70)

if valid_predictions_df.empty:
    print("没有有效预测样本进行置信度分析。")
else:
    print(f"\n整体置信度统计:")
    print(f"  平均置信度: {valid_predictions_df['confidence_score'].mean():.3f}")
    print(f"  置信度中位数: {valid_predictions_df['confidence_score'].median():.3f}")
    print(f"  置信度标准差: {valid_predictions_df['confidence_score'].std():.3f}")

    # 正确vs错误预测的置信度对比
    print(f"\n正确vs错误预测的置信度对比 (不含UNCRETRAIN):")
    correct_conf = valid_predictions_df[valid_predictions_df["is_correct"]][
        "confidence_score"
    ].mean()
    incorrect_conf = valid_predictions_df[~valid_predictions_df["is_correct"]][
        "confidence_score"
    ].mean()
    print(f"  正确预测平均置信度: {correct_conf:.3f}")
    print(f"  错误预测平均置信度: {incorrect_conf:.3f}")
    print(f"  差异 (越正越好): {correct_conf - incorrect_conf:.3f}")


# ===============================
# 新增：长序列 vs. 短序列性能比较
# ===============================
print("\n" + "=" * 70)
print("=== 序列长度性能比较 ===")
print("=" * 70)

if valid_predictions_df.empty:
    print("没有有效预测样本进行长度比较。")
else:
    # 计算测试集所有有效样本的平均长度
    avg_length = valid_predictions_df["sequence_length"].mean()
    print(f"测试集平均序列长度: {avg_length:.1f} 帧")

    # 划分长序列和短序列
    long_sequences = valid_predictions_df[
        valid_predictions_df["sequence_length"] > avg_length
    ]
    short_sequences = valid_predictions_df[
        valid_predictions_df["sequence_length"] <= avg_length
    ]

    print("\n--- 短序列 (长度 <= 平均) ---")
    if not short_sequences.empty:
        short_acc = short_sequences["is_correct"].mean()
        short_conf = short_sequences["confidence_score"].mean()
        print(f"  样本数: {len(short_sequences)}")
        print(f"  准确率: {short_acc:.3f}")
        print(f"  平均置信度: {short_conf:.3f}")
    else:
        print("  无样本。")

    print("\n--- 长序列 (长度 > 平均) ---")
    if not long_sequences.empty:
        long_acc = long_sequences["is_correct"].mean()
        long_conf = long_sequences["confidence_score"].mean()
        print(f"  样本数: {len(long_sequences)}")
        print(f"  准确率: {long_acc:.3f}")
        print(f"  平均置信度: {long_conf:.3f}")
    else:
        print("  无样本。")

    # 总结对比
    if not long_sequences.empty and not short_sequences.empty:
        acc_diff = long_acc - short_acc
        conf_diff = long_conf - short_conf
        print("\n--- 结论 ---")
        print(f"长序列 vs. 短序列 准确率差异 (长-短): {acc_diff:+.3f}")
        if abs(acc_diff) > 0.05:
            print("⚠️ 准确率差异较大，模型对序列长度存在偏好。")

        print(f"长序列 vs. 短序列 平均置信度差异 (长-短): {conf_diff:+.3f}")
        if conf_diff < -0.1:
            print("⚠️ 长序列的置信度显著偏低，可能意味着长度归一化不够充分。")
        elif conf_diff > 0.1:
            print("⚠️ 长序列的置信度显著偏高，可能存在过度自信。")

    # 确保保存的DF包含sequence_length
    confidence_df.to_csv("confidence_analysis_full.csv", index=False)
    print(f"\n💾 完整分析结果已保存至: confidence_analysis_full.csv")


# ===============================
# 低/高置信度样本分析 (基于有效预测)
# ===============================
if not valid_predictions_df.empty:

    # 低置信度样本
    low_conf_samples = valid_predictions_df[
        valid_predictions_df["confidence_score"] < CONFIDENCE_THRESHOLD_LOW
    ].sort_values("confidence_score")

    print("\n" + "=" * 70)
    print("=== 低置信度样本（需人工复核）===")
    print("=" * 70)
    print(
        f"\n发现 {len(low_conf_samples)} 个低置信度样本 (阈值 < {CONFIDENCE_THRESHOLD_LOW}):"
    )

    if len(low_conf_samples) > 0:
        print(
            f"\n{'视频ID':<30} {'真实':<10} {'预测':<10} {'长度':<8} {'置信度':<10} {'正确?'}"
        )
        print("-" * 80)
        for _, row in low_conf_samples.head(10).iterrows():
            check = "✓" if row["is_correct"] else "✗"
            print(
                f"{row['video_id']:<30} {row['true_label']:<10} {row['predicted_label']:<10} "
                f"{row['sequence_length']:<8} {row['confidence_score']:<10.3f} {check}"
            )

        low_conf_file = "low_confidence_samples.csv"
        low_conf_samples.to_csv(low_conf_file, index=False)
        print(f"\n💾 低置信度样本已保存至: {low_conf_file}")

    # 高置信度错误样本
    high_conf_errors = valid_predictions_df[
        (~valid_predictions_df["is_correct"])
        & (valid_predictions_df["confidence_score"] > CONFIDENCE_THRESHOLD_HIGH)
    ].sort_values("confidence_score", ascending=False)

    print("\n" + "=" * 70)
    print("=== 高置信度但预测错误的样本（重点！）===")
    print("=" * 70)

    if len(high_conf_errors) > 0:
        print(
            f"\n发现 {len(high_conf_errors)} 个高置信度错误样本 (置信度 > {CONFIDENCE_THRESHOLD_HIGH}):"
        )
        print("这些样本可能存在标签错误或特征提取问题！\n")

        print(f"{'视频ID':<30} {'真实':<10} {'预测':<10} {'长度':<8} {'置信度':<10}")
        print("-" * 80)
        for _, row in high_conf_errors.head(10).iterrows():
            print(
                f"{row['video_id']:<30} {row['true_label']:<10} {row['predicted_label']:<10} "
                f"{row['sequence_length']:<8} {row['confidence_score']:<10.3f}"
            )

        high_conf_errors.to_csv("high_confidence_errors.csv", index=False)
        print(f"\n💾 高置信度错误样本已保存至: high_confidence_errors.csv")
    else:
        print("\n未发现高置信度错误样本。")

print("\n" + "=" * 70)
print("分析完成！")
print("=" * 70)
