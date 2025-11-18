"""
置信度分析可视化脚本
需要先运行主脚本生成 confidence_analysis_full.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体（根据系统调整）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取分析结果
try:
    df = pd.read_csv('confidence_analysis_full.csv')
    print(f"成功加载 {len(df)} 条记录")
except FileNotFoundError:
    print("错误: 请先运行主脚本生成 confidence_analysis_full.csv")
    exit()

# 设置绘图风格
sns.set_style("whitegrid")
plt.figure(figsize=(20, 12))

# ===============================
# 图1: 置信度分布直方图
# ===============================
plt.subplot(2, 3, 1)
plt.hist(df['confidence_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(df['confidence_score'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["confidence_score"].mean():.3f}')
plt.axvline(df['confidence_score'].median(), color='green', linestyle='--', 
            label=f'Median: {df["confidence_score"].median():.3f}')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Confidence Score Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# ===============================
# 图2: 正确vs错误预测的置信度对比
# ===============================
plt.subplot(2, 3, 2)
correct_conf = df[df['is_correct']]['confidence_score']
incorrect_conf = df[~df['is_correct']]['confidence_score']

plt.boxplot([correct_conf, incorrect_conf], 
            labels=['Correct', 'Incorrect'],
            patch_artist=True,
            boxprops=dict(facecolor='lightgreen', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
plt.ylabel('Confidence Score')
plt.title('Confidence: Correct vs Incorrect Predictions')
plt.grid(True, alpha=0.3)

# 添加统计信息
plt.text(0.5, 0.95, f'Correct: μ={correct_conf.mean():.3f}', 
         transform=plt.gca().transAxes, fontsize=9)
plt.text(0.5, 0.90, f'Incorrect: μ={incorrect_conf.mean():.3f}', 
         transform=plt.gca().transAxes, fontsize=9)

# ===============================
# 图3: 各类别准确率
# ===============================
plt.subplot(2, 3, 3)
class_accuracy = df.groupby('true_label').apply(
    lambda x: (x['is_correct'].sum() / len(x)) * 100
).sort_values()

colors = ['red' if acc < 60 else 'orange' if acc < 80 else 'green' 
          for acc in class_accuracy.values]

class_accuracy.plot(kind='barh', color=colors, alpha=0.7)
plt.xlabel('Accuracy (%)')
plt.ylabel('Class')
plt.title('Per-Class Accuracy')
plt.axvline(80, color='green', linestyle='--', alpha=0.5, label='Good (>80%)')
plt.axvline(60, color='orange', linestyle='--', alpha=0.5, label='Fair (>60%)')
plt.legend()
plt.grid(True, alpha=0.3)

# ===============================
# 图4: 各类别平均置信度
# ===============================
plt.subplot(2, 3, 4)
class_confidence = df.groupby('true_label')['confidence_score'].mean().sort_values()
class_confidence.plot(kind='barh', color='steelblue', alpha=0.7)
plt.xlabel('Average Confidence')
plt.ylabel('Class')
plt.title('Per-Class Average Confidence')
plt.grid(True, alpha=0.3)

# ===============================
# 图5: 置信度 vs 准确率（散点图）
# ===============================
plt.subplot(2, 3, 5)
conf_bins = pd.cut(df['confidence_score'], bins=10)
accuracy_by_conf = df.groupby(conf_bins).apply(
    lambda x: (x['is_correct'].sum() / len(x)) * 100 if len(x) > 0 else 0
)
count_by_conf = df.groupby(conf_bins).size()

x_positions = [interval.mid for interval in accuracy_by_conf.index]
plt.scatter(x_positions, accuracy_by_conf.values, 
           s=count_by_conf.values * 10, alpha=0.6, color='purple')
plt.plot(x_positions, accuracy_by_conf.values, 'r--', alpha=0.5)
plt.xlabel('Confidence Score')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Confidence (bubble size = sample count)')
plt.grid(True, alpha=0.3)

# ===============================
# 图6: Margin分布
# ===============================
plt.subplot(2, 3, 6)
plt.hist(df['margin'], bins=30, alpha=0.7, color='coral', edgecolor='black')
plt.axvline(df['margin'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["margin"].mean():.2f}')
plt.xlabel('Margin (best - 2nd best score)')
plt.ylabel('Frequency')
plt.title('Margin Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('confidence_analysis_visualization.png', dpi=300, bbox_inches='tight')
print("\n✅ 可视化图表已保存为: confidence_analysis_visualization.png")

# ===============================
# 额外图表: 混淆矩阵热图
# ===============================
plt.figure(figsize=(10, 8))
from sklearn.metrics import confusion_matrix

labels = sorted(df['true_label'].unique())
cm = confusion_matrix(df['true_label'], df['predicted_label'], labels=labels)

# 计算百分比
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='YlOrRd', 
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Percentage (%)'})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Percentage)')
plt.tight_layout()
plt.savefig('confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ 混淆矩阵热图已保存为: confusion_matrix_heatmap.png")

# ===============================
# 类别详细分析图
# ===============================
unique_labels = sorted(df['true_label'].unique())
n_labels = len(unique_labels)

fig, axes = plt.subplots(n_labels, 2, figsize=(15, 4*n_labels))
if n_labels == 1:
    axes = axes.reshape(1, -1)

for idx, label in enumerate(unique_labels):
    label_data = df[df['true_label'] == label]
    
    # 左图: 该类别的置信度分布
    ax1 = axes[idx, 0]
    correct_data = label_data[label_data['is_correct']]['confidence_score']
    incorrect_data = label_data[~label_data['is_correct']]['confidence_score']
    
    ax1.hist([correct_data, incorrect_data], bins=20, alpha=0.7,
             label=['Correct', 'Incorrect'], color=['green', 'red'])
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Class: {label} - Confidence Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图: 该类别被误分类为哪些类别
    ax2 = axes[idx, 1]
    confused = label_data[~label_data['is_correct']]['predicted_label'].value_counts()
    if len(confused) > 0:
        confused.plot(kind='barh', ax=ax2, color='orange', alpha=0.7)
        ax2.set_xlabel('Count')
        ax2.set_title(f'Class: {label} - Confused As')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No misclassifications!', 
                ha='center', va='center', fontsize=14, color='green')
        ax2.set_title(f'Class: {label} - Confused As')

plt.tight_layout()
plt.savefig('per_class_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 各类别详细分析已保存为: per_class_analysis.png")

# ===============================
# 生成文本报告
# ===============================
report = []
report.append("=" * 70)
report.append("置信度分析报告")
report.append("=" * 70)
report.append("")

# 整体统计
report.append("📊 整体统计:")
report.append(f"  总样本数: {len(df)}")
report.append(f"  整体准确率: {(df['is_correct'].sum() / len(df)) * 100:.2f}%")
report.append(f"  平均置信度: {df['confidence_score'].mean():.3f}")
report.append(f"  置信度标准差: {df['confidence_score'].std():.3f}")
report.append("")

# 按置信度区间统计
report.append("📈 按置信度区间统计:")
bins = [0, 0.3, 0.5, 0.7, 1.0]
labels_bin = ['Low (0-0.3)', 'Medium-Low (0.3-0.5)', 'Medium-High (0.5-0.7)', 'High (0.7-1.0)']
df['conf_bin'] = pd.cut(df['confidence_score'], bins=bins, labels=labels_bin)

for bin_label in labels_bin:
    bin_data = df[df['conf_bin'] == bin_label]
    if len(bin_data) > 0:
        acc = (bin_data['is_correct'].sum() / len(bin_data)) * 100
        report.append(f"  {bin_label}: {len(bin_data)} 样本, 准确率 {acc:.1f}%")
report.append("")

# 各类别表现
report.append("🎯 各类别表现:")
for label in sorted(df['true_label'].unique()):
    label_data = df[df['true_label'] == label]
    acc = (label_data['is_correct'].sum() / len(label_data)) * 100
    avg_conf = label_data['confidence_score'].mean()
    report.append(f"  {label}:")
    report.append(f"    样本数: {len(label_data)}")
    report.append(f"    准确率: {acc:.2f}%")
    report.append(f"    平均置信度: {avg_conf:.3f}")
report.append("")

# 问题样本
low_conf_count = len(df[df['confidence_score'] < 0.3])
high_conf_errors = len(df[(~df['is_correct']) & (df['confidence_score'] > 0.7)])

report.append("⚠️ 需要关注的样本:")
report.append(f"  低置信度样本 (<0.3): {low_conf_count}")
report.append(f"  高置信度错误 (>0.7): {high_conf_errors}")
report.append("")

# 保存报告
with open('confidence_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print("✅ 文本报告已保存为: confidence_analysis_report.txt")
print("\n" + "=" * 70)
print("可视化分析完成！")
print("=" * 70)
print("\n生成的文件:")
print("  1. confidence_analysis_visualization.png - 6个综合分析图")
print("  2. confusion_matrix_heatmap.png - 混淆矩阵热图")
print("  3. per_class_analysis.png - 各类别详细分析")
print("  4. confidence_analysis_report.txt - 文本分析报告")
print("\n建议查看顺序:")
print("  1. 先看综合分析图，了解整体情况")
print("  2. 查看混淆矩阵，找出容易混淆的类别对")
print("  3. 看各类别详细分析，针对性改进")
print("=" * 70)