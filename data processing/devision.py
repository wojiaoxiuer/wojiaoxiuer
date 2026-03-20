import pandas as pd
from pathlib import Path
import shutil

# 配置路径
excel_path = "train_path.xlsx"
src_root = Path("Bra21_train2d_jpg")
dst_root = Path("Bra21")

# 读取Excel建立映射关系
df = pd.read_excel(excel_path, header=None)
folder_mapping = {}
for _, row in df.iterrows():
    label = "positive" if row[0] == 1 else "negative"
    short_name = str(row[1])
    # 将短名称转换为5位数字格式
    full_name = short_name.zfill(5)
    folder_mapping[full_name] = label

# 遍历源目录处理文件
for src_folder in src_root.iterdir():
    if src_folder.is_dir():
        folder_name = src_folder.name

        # 验证是否为5位数字文件夹
        if not (folder_name.isdigit() and len(folder_name) == 5):
            continue

        # 获取分类信息
        if folder_name not in folder_mapping:
            print(f"警告：未找到映射关系的文件夹 {folder_name}")
            continue

        category = folder_mapping[folder_name]

        # 处理每个图像文件
        for img_file in src_folder.glob("*.jpg"):
            # 提取模态类型
            modality = img_file.stem  # 获取不带扩展名的文件名

            # 构建新文件名
            new_filename = f"{folder_name}{modality}.jpg"

            # 构建目标路径
            dest_dir = dst_root / category / modality
            dest_dir.mkdir(parents=True, exist_ok=True)

            # 复制文件（保留原文件）
            shutil.copy(img_file, dest_dir / new_filename)
            print(f"已复制：{img_file} -> {dest_dir / new_filename}")

print("所有文件处理完成！")