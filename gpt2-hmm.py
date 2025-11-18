"""
HMM超参数网格搜索脚本
用于系统地测试不同参数组合，找到最佳precision
"""
# 参数搜索范围
# N_COMPONENTS_RANGE = [2, 3, 4, 5]  # 隐状态数
# N_MIXTURES_RANGE = [2, 3, 4]  # 高斯混合数
# COV_TYPES = ["diag", "full"]  # 协方差类型
# MAX_ITER_RANGE = [100, 200]  # 最大迭代次数
# 改进意见： 特征优化，角度的平方        数据平衡处理，类别平衡！

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from hmmlearn.hmm import GMMHMM
from itertools import product
import time

# ===============================
# 基础配置
# ===============================
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "demobb")
LABEL_FILE = os.path.join(os.path.dirname(__file__), "labels.csv")
RANDOM_STATE = 42
NAN_STRATEGY = "interpolate"
MIN_VALID_FRAMES = 10

np.random.seed(RANDOM_STATE)

# ===============================
# 超参数搜索空间
# ===============================
PARAM_GRID = {
    'n_components': [3, 4, 5, 6],           # 隐状态数
    'n_mixtures': [1, 2, 3],                # GMM混合数
    'covariance_type': ['diag', 'full'],    # 协方差类型
    'max_iter': [100],                       # EM迭代次数（可固定）
}

# 快速测试配置（用于快速验证）
PARAM_GRID_FAST = {
    'n_components': [3, 4, 5],
    'n_mixtures': [1, 2],
    'covariance_type': ['diag'],
    'max_iter': [50],
}

# 选择搜索模式
USE_FAST_MODE = False  # True=快速模式, False=完整搜索

# ===============================
# 数据加载函数（同原脚本）
# ===============================
def interpolate_nan(seq):
    df = pd.DataFrame(seq)
    df = df.interpolate(method='linear', limit_direction='both', axis=0)
    df = df.fillna(df.mean())
    df = df.fillna(0)
    return df.values

def drop_nan_frames(seq):
    mask = ~np.isnan(seq).any(axis=1)
    return seq[mask]

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
    #     # 添加角度的平方作为特征
    # angles_squared = angles ** 2
    # seq.append(np.concatenate([angles, angles_squared]))
    # return np.array(seq)
    # 角度权重增加
    # if label == "jg":
    #     weights = [1.0, 1.0, 1.0, 1.0, 2.0]  # 对jg类别，ankle角度权重加倍
    # else:
    #     weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # 其他类别保持原始权重
    # # 应用权重
    # seq = seq * weights

    if np.isnan(seq).any():
        if NAN_STRATEGY == "interpolate":
            seq = interpolate_nan(seq)
        elif NAN_STRATEGY == "drop":
            seq = drop_nan_frames(seq)

    return seq

def is_valid_sequence(seq):
    if len(seq) < MIN_VALID_FRAMES:
        return False
    if np.isnan(seq).all():
        return False
    return True

# ===============================
# 加载数据
# ===============================
print("加载数据...")
labels_df = pd.read_csv(LABEL_FILE)
sequences, labels = [], []

for _, row in labels_df.iterrows():
    json_path = os.path.join(DATA_FOLDER, row["video_id"])
    if os.path.exists(json_path):
        try:
            seq = load_json_angles(json_path)
            if is_valid_sequence(seq):
                sequences.append(seq)
                labels.append(row["label"])
        except Exception as e:
            continue

print(f"成功加载 {len(sequences)} 个样本")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
)

# 特征标准化
scaler = StandardScaler()
all_train_frames = np.vstack(X_train)
scaler.fit(all_train_frames)
X_train = [scaler.transform(x) for x in X_train]
X_test = [scaler.transform(x) for x in X_test]

print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ===============================
# 训练与评估函数
# ===============================
def train_and_evaluate(n_components, n_mixtures, covariance_type, max_iter):
    """
    训练HMM模型并返回评估指标
    """
    models = {}
    unique_labels = sorted(set(y_train))
    
    try:
        # 训练每类模型
        for label in unique_labels:
            label_seqs = [X_train[i] for i in range(len(X_train)) if y_train[i] == label]
            X_concat = np.vstack(label_seqs)
            lengths = [len(x) for x in label_seqs]
            
            model = GMMHMM(
                n_components=n_components,
                n_mix=n_mixtures,
                covariance_type=covariance_type,
                n_iter=max_iter,
                random_state=RANDOM_STATE,
                verbose=False,
            )
            model.fit(X_concat, lengths)
            models[label] = model
        
        # 预测
        y_pred = []
        for seq in X_test:
            logL = {label: models[label].score(seq) for label in models}
            pred_label = max(logL, key=logL.get)
            y_pred.append(pred_label)
        
        # 计算指标（macro平均）
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        return accuracy, precision, recall, f1, True
    
    except Exception as e:
        print(f"  训练失败: {e}")
        return 0, 0, 0, 0, False

# ===============================
# 网格搜索
# ===============================
param_grid = PARAM_GRID_FAST if USE_FAST_MODE else PARAM_GRID

# 生成所有参数组合
param_combinations = list(product(
    param_grid['n_components'],
    param_grid['n_mixtures'],
    param_grid['covariance_type'],
    param_grid['max_iter']
))

print(f"\n开始网格搜索 ({'快速模式' if USE_FAST_MODE else '完整模式'})")
print(f"总共 {len(param_combinations)} 种参数组合\n")
print("="*80)

results = []
best_precision = 0
best_params = None

for idx, (n_comp, n_mix, cov_type, max_iter) in enumerate(param_combinations, 1):
    print(f"\n[{idx}/{len(param_combinations)}] 测试参数:")
    print(f"  n_components={n_comp}, n_mixtures={n_mix}, cov_type={cov_type}, max_iter={max_iter}")
    
    start_time = time.time()
    acc, prec, rec, f1, success = train_and_evaluate(n_comp, n_mix, cov_type, max_iter)
    elapsed = time.time() - start_time
    
    if success:
        print(f"  ✅ Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f} (耗时: {elapsed:.1f}s)")
        
        results.append({
            'n_components': n_comp,
            'n_mixtures': n_mix,
            'covariance_type': cov_type,
            'max_iter': max_iter,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'time': elapsed
        })
        
        # 更新最佳结果
        if prec > best_precision:
            best_precision = prec
            best_params = (n_comp, n_mix, cov_type, max_iter)
            print(f"  🌟 新的最佳Precision!")

# ===============================
# 输出结果
# ===============================
print("\n" + "="*80)
print("=== 调参结果汇总 ===")
print("="*80)

# 转换为DataFrame
results_df = pd.DataFrame(results)

# 按precision排序
results_df = results_df.sort_values('precision', ascending=False)

print("\n前10个最佳配置（按Precision排序）:")
print(results_df.head(10).to_string(index=False))

print(f"\n最佳参数组合:")
print(f"  n_components: {best_params[0]}")
print(f"  n_mixtures: {best_params[1]}")
print(f"  covariance_type: {best_params[2]}")
print(f"  max_iter: {best_params[3]}")
print(f"  → 最佳Precision: {best_precision:.4f}")

# 保存结果
output_file = "hmm_tuning_results.csv"
results_df.to_csv(output_file, index=False)
print(f"\n完整结果已保存至: {output_file}")

# ===============================
# 参数影响分析
# ===============================
print("\n" + "="*80)
print("=== 参数影响分析 ===")
print("="*80)

if len(results_df) > 0:
    # 按各个参数分组，查看平均precision
    print("\n各参数对Precision的影响:")
    
    for param in ['n_components', 'n_mixtures', 'covariance_type']:
        print(f"\n{param}:")
        grouped = results_df.groupby(param)['precision'].agg(['mean', 'std', 'max'])
        print(grouped.round(4))

print("\n" + "="*80)
print("调参建议:")
print("1. 如果所有配置precision都很低(<0.5)，考虑:")
print("   - 检查数据质量和标签是否正确")
print("   - 增加训练样本数量")
print("   - 尝试不同的特征工程方法")
print("\n2. 如果某个参数明显更优，固定该参数后继续细化其他参数")
print("\n3. 如果covariance_type='full'效果好但训练慢，考虑增加样本后再用")
print("="*80)
