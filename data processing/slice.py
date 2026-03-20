import numpy as np
from pathlib import Path
import os


original_dir = 'Bra21gg'
new_dir = 'Bra21_train2d'


for file_path in Path(original_dir).rglob('*.npy'):
    
    relative_path = file_path.relative_to(original_dir)
    new_file_path = Path(new_dir) / relative_path

   
    new_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载3D数据
    data_3d = np.load(str(file_path))

   
    
    slice_intensities = data_3d.sum(axis=(1, 2)) 
    
    # slice_intensities = data_3d.max(axis=(1, 2)) 

    max_slice_idx = np.argmax(slice_intensities)
    max_slice = data_3d[max_slice_idx]  # 提取该切片

    
    np.save(str(new_file_path), max_slice)

print("sucessful！")
