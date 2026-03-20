import numpy as np
from pathlib import Path
import os

# 设置原始目录和新目录路径
original_dir = 'Bra21gg'
new_dir = 'Bra21_train2d'

# 遍历原始目录中的所有.npy文件
for file_path in Path(original_dir).rglob('*.npy'):
    # 获取相对路径并创建新路径
    relative_path = file_path.relative_to(original_dir)
    new_file_path = Path(new_dir) / relative_path

    # 创建目标目录（如果不存在）
    new_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载3D数据（假设形状为 [depth, height, width]）
    data_3d = np.load(str(file_path))

    # 寻找最大强度切片
    # 方法1：通过切片总强度寻找
    slice_intensities = data_3d.sum(axis=(1, 2))  # 计算每个切片的总强度
    # 方法2：通过切片最大强度寻找（二选一）
    # slice_intensities = data_3d.max(axis=(1, 2))  # 计算每个切片的最大强度

    max_slice_idx = np.argmax(slice_intensities)
    max_slice = data_3d[max_slice_idx]  # 提取该切片

    # 保存处理后的2D数据
    np.save(str(new_file_path), max_slice)

print("所有文件处理完成！")