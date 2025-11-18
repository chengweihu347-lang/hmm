import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from hmmlearn import hmm
import warnings

# 忽略hmmlearn库中关于GMMHMM API变更的未来警告
warnings.filterwarnings("ignore", category=FutureWarning)
# 忽略由于数据量少或模型复杂导致的拟合警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 数据预处理 ---


def load_sequence_from_json(json_path):
    """
    从单个JSON文件中加载动作序列数据。
    返回一个 (n_frames, n_features) 的 numpy 数组。
    处理None值，将其转换为np.nan。
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        sequence = []
        for frame in data["frames"]:
            # 提取5个关键角度作为特征
            joints = frame["joints"]
            features = [
                joints["shoulder_angle"],
                joints["elbow_angle"],
                joints["hip_angle"],
                joints["knee_angle"],
                joints["ankle_angle"],
            ]
            # 将None转换为np.nan，确保数据类型一致
            features = [float(f) if f is not None else np.nan for f in features]
            sequence.append(features)

        # 显式指定dtype为float，确保能正确处理NaN
        return np.array(sequence, dtype=np.float64)

    except Exception as e:
        print(f"错误：加载文件 {json_path}失败: {e}")
        return None


def check_and_handle_nan(sequence, method="interpolate"):
    """
    检查并处理序列中的NaN值。

    参数:
        sequence: numpy数组 (n_frames, n_features)
        method: 处理方法
            - 'interpolate': 线性插值
            - 'mean': 用特征均值填充
            - 'forward': 前向填充
            - 'remove': 移除含NaN的帧

    返回:
        处理后的序列和统计信息
    """
    if sequence is None or len(sequence) == 0:
        return None, {"total_nan": 0, "frames_affected": 0}

    # 统计NaN
    nan_mask = np.isnan(sequence)
    total_nan = np.sum(nan_mask)
    frames_with_nan = np.any(nan_mask, axis=1).sum()

    stats = {
        "total_nan": total_nan,
        "frames_affected": frames_with_nan,
        "original_length": len(sequence),
    }

    if total_nan == 0:
        return sequence, stats

    # 处理NaN
    if method == "remove":
        # 移除含有NaN的帧
        clean_sequence = sequence[~np.any(nan_mask, axis=1)]
        stats["final_length"] = len(clean_sequence)
        return clean_sequence, stats

    elif method == "interpolate":
        # 使用pandas进行线性插值
        df = pd.DataFrame(sequence)
        df_interpolated = df.interpolate(method="linear", limit_direction="both")
        # 如果首尾仍有NaN，用前向/后向填充
        df_interpolated = df_interpolated.fillna(method="ffill").fillna(method="bfill")
        stats["final_length"] = len(df_interpolated)
        return df_interpolated.values, stats

    elif method == "mean":
        # 用每个特征的均值填充
        imputer = SimpleImputer(strategy="mean")
        clean_sequence = imputer.fit_transform(sequence)
        stats["final_length"] = len(clean_sequence)
        return clean_sequence, stats

    elif method == "forward":
        # 前向填充，然后后向填充（处理开头的NaN）
        df = pd.DataFrame(sequence)
        df_filled = df.fillna(method="ffill").fillna(method="bfill")
        stats["final_length"] = len(df_filled)
        return df_filled.values, stats

    else:
        raise ValueError(f"未知的NaN处理方法: {method}")


def preprocess_data(
    data_folder, label_file_path, nan_method="interpolate", min_sequence_length=10
):
    """
    加载所有数据，按标签分组，并进行特征标准化。

    参数:
        data_folder: 数据文件夹路径
        label_file_path: 标签文件路径
        nan_method: NaN处理方法
        min_sequence_length: 最小序列长度（过滤太短的序列）
    """
    print("开始数据预处理...")
    print(f"NaN处理方法: {nan_method}")
    labels_df = pd.read_csv(label_file_path)

    sequences = {}  # 存储每个标签对应的所有序列
    all_sequences = []  # 存储所有序列，用于训练Scaler
    nan_statistics = []  # 存储NaN统计信息

    for _, row in labels_df.iterrows():
        video_id = row["video_id"]
        label = row["label"]

        # 假设demobb在data_folder中
        json_path = os.path.join(data_folder, "demobb", video_id)

        seq = load_sequence_from_json(json_path)

        if seq is not None and len(seq) > 0:
            # 处理NaN值
            clean_seq, nan_stats = check_and_handle_nan(seq, method=nan_method)

            # 记录统计信息
            if nan_stats["total_nan"] > 0:
                nan_statistics.append(
                    {"video_id": video_id, "label": label, **nan_stats}
                )
                print(
                    f"  {video_id}: 发现 {nan_stats['total_nan']} 个NaN值, "
                    f"影响 {nan_stats['frames_affected']} 帧"
                )

            # 检查处理后的序列
            if clean_seq is not None and len(clean_seq) >= min_sequence_length:
                # 再次检查是否还有NaN（双重保险）
                if np.any(np.isnan(clean_seq)):
                    print(f"警告：{video_id} 处理后仍包含NaN，跳过该序列")
                    continue

                if label not in sequences:
                    sequences[label] = []

                sequences[label].append(clean_seq)
                all_sequences.append(clean_seq)
            else:
                print(f"警告：跳过序列 {video_id} (长度不足或处理失败)")
        else:
            print(f"警告：跳过空序列或加载失败的文件 {video_id}")

    # 打印NaN统计摘要
    if nan_statistics:
        print(f"\n=== NaN统计摘要 ===")
        print(f"受影响的序列数: {len(nan_statistics)}")
        total_nan_count = sum(s["total_nan"] for s in nan_statistics)
        print(f"总NaN值数量: {total_nan_count}")
    else:
        print("\n✓ 所有序列都没有NaN值")

    # --- 特征标准化 (GMM的关键步骤) ---
    if not all_sequences:
        raise ValueError("没有成功加载任何序列数据！")

    concatenated_data = np.concatenate(all_sequences, axis=0)

    # 最后检查：确保标准化前数据没有NaN
    if np.any(np.isnan(concatenated_data)):
        print("警告：标准化前发现NaN值，进行最终清理...")
        imputer = SimpleImputer(strategy="mean")
        concatenated_data = imputer.fit_transform(concatenated_data)

    scaler = StandardScaler()
    scaler.fit(concatenated_data)

    # 对每个序列应用标准化
    scaled_sequences = {}
    for label, seq_list in sequences.items():
        scaled_sequences[label] = [scaler.transform(seq) for seq in seq_list]

    print(f"\n数据预处理完成。")
    print(f"成功加载的标签数: {len(scaled_sequences)}")
    for label, seq_list in scaled_sequences.items():
        print(f"  {label}: {len(seq_list)} 个序列")

    return scaled_sequences, scaler


# --- 2. HMM 模型训练 (Baum-Welch) ---


def train_hmm_models(scaled_sequences, hmm_params):
    """
    为每个类别训练一个GMM-HMM模型。
    """
    print("\n开始训练HMM模型...")
    models = {}

    for label, seq_list in scaled_sequences.items():
        if not seq_list:
            print(f"警告：类别 {label} 没有任何数据，跳过训练。")
            continue

        print(f"  正在训练模型 for: {label}")

        # 准备数据：hmmlearn需要一个大的连接数组和每个序列的长度
        X = np.concatenate(seq_list, axis=0)
        lengths = [len(seq) for seq in seq_list]

        # 训练前最终检查
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print(f"!! 错误：{label} 的训练数据包含NaN或Inf值，跳过训练")
            continue

        # 初始化GMM-HMM
        model = hmm.GMMHMM(
            n_components=hmm_params["n_components"],
            n_mix=hmm_params["n_mix"],
            covariance_type=hmm_params["cov_type"],
            n_iter=hmm_params["n_iter"],
            tol=1e-4,
            random_state=42,
            verbose=False,
            params="stmcw",
            init_params="stmcw",
        )

        try:
            # 使用 Baum-Welch 算法 (EM)进行训练
            model.fit(X, lengths)
            models[label] = model
            print(f"  模型 {label} 训练完成。")
        except ValueError as e:
            print(f"!! 训练 {label} 模型失败: {e}")
            print("   这通常发生在数据量太少或n_components/n_mix过高时。")

    print("\n所有HMM模型训练完毕。")
    return models


# --- 3. 打印模型参数 ---


def print_model_parameters(models):
    """
    打印训练好的模型的关键参数。
    """
    for label, model in models.items():
        print(f"\n--- {label} 模型的参数 ---")

        if not model.monitor_.converged:
            print("!! 警告：模型训练未收敛 !!")

        # 1. 初始状态概率 (Pi)
        print("\n[初始状态矩阵 (Pi)] (维度: 1 x n_components)")
        print(model.startprob_)

        # 2. 状态转移矩阵 (A)
        print("\n[状态转移矩阵 (A)] (维度: n_components x n_components)")
        print(model.transmat_)

        # 3. 观测概率 (B) - GMM-HMM
        print("\n[观测概率 (B) - 由GMM参数表示]")
        print("GMM-HMM的'B'不是一个简单的矩阵。")
        print("它由每个状态下 GMM 的 (权重, 均值, 协方差) 决定。")

        for i in range(model.n_components):
            print(f"  [状态 {i} 的 GMM 参数]")
            print(f"    GMM 权重 (n_mix): {model.weights_[i]}")
            print(f"    GMM 均值 (n_mix x n_features):\n{model.means_[i]}")
            print(f"    GMM 协方差形状: {model.covars_[i].shape}")


# --- 4. 预测与Viterbi算法 ---


def classify_new_sequence(new_json_path, models, scaler, nan_method="interpolate"):
    """
    使用训练好的模型对新序列进行分类。
    """
    print(f"\n--- 正在预测新文件: {new_json_path} ---")

    # 1. 加载和预处理新序列
    new_seq = load_sequence_from_json(new_json_path)
    if new_seq is None or len(new_seq) == 0:
        print("无法加载或序列为空，预测失败。")
        return None, None

    # 处理NaN值
    clean_seq, nan_stats = check_and_handle_nan(new_seq, method=nan_method)

    if nan_stats["total_nan"] > 0:
        print(f"  发现并处理了 {nan_stats['total_nan']} 个NaN值")

    if clean_seq is None or len(clean_seq) == 0:
        print("序列处理后为空，预测失败。")
        return None, None

    # 最终检查
    if np.any(np.isnan(clean_seq)):
        print("错误：序列处理后仍包含NaN值，预测失败。")
        return None, None

    scaled_seq = scaler.transform(clean_seq)

    # 2. 计算每个模型的得分（对数似然）
    best_score = -np.inf
    predicted_label = None
    log_scores = {}

    for label, model in models.items():
        try:
            score = model.score(scaled_seq)
            log_scores[label] = score

            if score > best_score:
                best_score = score
                predicted_label = label
        except Exception as e:
            print(f"计算 {label} 模型得分时出错: {e}")
            log_scores[label] = -np.inf

    print("\n所有模型的对数似然 (Log-Likelihood) 得分:")
    for label, score in log_scores.items():
        print(f"  {label}: {score:.2f}")

    print(f"\n==> 预测结果: {predicted_label} (得分: {best_score:.2f})")

    # 3. Viterbi 算法演示
    viterbi_path = None
    if predicted_label:
        best_model = models[predicted_label]

        # Viterbi 解码
        log_prob, viterbi_path = best_model.decode(scaled_seq, algorithm="viterbi")

        print(f"\n[Viterbi 算法演示 (针对 {predicted_label} 模型)]")
        print(f"  Viterbi路径的对数概率: {log_prob:.2f}")
        print(f"  最可能的隐状态序列 (前20帧): \n{viterbi_path[:20]}...")
        print(
            f"  (注意：这些是 {predicted_label} 模型的内部隐状态, 比如 0, 1, 2... 而不是 'bb', 'jg')"
        )

    return predicted_label, viterbi_path


# --- 5. 主执行函数 ---


def main():
    # --- !! 需要您手动设置的路径 !! ---
    BASE_DIR = "."
    DATA_FOLDER = BASE_DIR
    LABEL_FILE = os.path.join(BASE_DIR, "labels.csv")

    # --- !! 需要手动调节的超参数 !! ---
    hmm_hyperparams = {
        "n_components": 3,  # 每个HMM的隐状态数量
        "n_mix": 2,  # 每个状态的 GMM 组件数量
        "cov_type": "diag",  # 协方差类型
        "n_iter": 50,  # Baum-Welch 训练迭代次数
    }

    # NaN处理配置
    nan_config = {
        "method": "interpolate",  # 可选: 'interpolate', 'mean', 'forward', 'remove'
        "min_sequence_length": 10,  # 最小序列长度
    }

    # 1. 加载和预处理数据
    scaled_data, scaler = preprocess_data(
        DATA_FOLDER,
        LABEL_FILE,
        nan_method=nan_config["method"],
        min_sequence_length=nan_config["min_sequence_length"],
    )

    # 2. 训练模型
    models = train_hmm_models(scaled_data, hmm_hyperparams)

    # 3. 打印模型参数
    print_model_parameters(models)

    # 4. 预测示例
    TEST_JSON_FILE = os.path.join(DATA_FOLDER, "demobb", "bb1_angles.json")

    if os.path.exists(TEST_JSON_FILE):
        classify_new_sequence(
            TEST_JSON_FILE, models, scaler, nan_method=nan_config["method"]
        )
    else:
        print(f"\n未找到测试文件 {TEST_JSON_FILE}，跳过预测。")
        print("请将 'TEST_JSON_FILE' 变量修改为您的新JSON文件路径。")


if __name__ == "__main__":
    main()
