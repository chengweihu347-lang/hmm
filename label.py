import os
import csv
import re


def extract_label_from_filename(filename):
    """
    从文件名中提取标签
    例如: bb_angles.json -> bb
         bb2_angles.json -> bb
         jg_angles.json -> jg
         stand3_angles.json -> stand
    """
    # 移除 .json 后缀
    name_without_ext = filename.replace(".json", "")

    # 使用正则表达式匹配模式：标签名 + 可选数字 + _angles
    # 匹配 bb, bb2, jg, jg2, stand, stand2 等
    match = re.match(r"^([a-zA-Z]+)\d*_angles$", name_without_ext)

    if match:
        return match.group(1)  # 返回标签部分（不含数字）
    else:
        return None


def generate_labels_csv(folder_path="demobb", output_file="labels.csv"):
    """
    扫描指定文件夹中的JSON文件并生成labels.csv

    参数:
        folder_path: JSON文件所在的文件夹路径，默认为'demobb'
        output_file: 输出的CSV文件名，默认为'labels.csv'
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在！")
        return

    # 存储结果的列表
    labels_data = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理JSON文件
        if filename.endswith(".json") and "_angles" in filename:
            label = extract_label_from_filename(filename)

            if label:
                labels_data.append({"video_id": filename, "label": label})
                print(f"处理: {filename} -> 标签: {label}")
            else:
                print(f"警告: 无法从 '{filename}' 提取标签")

    # 按video_id排序
    labels_data.sort(key=lambda x: x["video_id"])

    # 写入CSV文件
    if labels_data:
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["video_id", "label"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(labels_data)

        print(f"\n成功! 已生成 '{output_file}'，共 {len(labels_data)} 条记录")
    else:
        print("\n警告: 未找到符合条件的JSON文件")


if __name__ == "__main__":
    # 运行脚本
    generate_labels_csv()

    # 如果你的文件夹路径不同，可以这样调用：
    # generate_labels_csv(folder_path='your_folder_name', output_file='labels.csv')
