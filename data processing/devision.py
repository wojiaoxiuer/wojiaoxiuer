import pandas as pd
from pathlib import Path
import shutil


excel_path = "excel_path"
src_root = Path("data_path")
dst_root = Path("data_path")


df = pd.read_excel(excel_path, header=None)
folder_mapping = {}
for _, row in df.iterrows():
    label = "positive" if row[0] == 1 else "negative"
    short_name = str(row[1])
    
    full_name = short_name.zfill(5)
    folder_mapping[full_name] = label


for src_folder in src_root.iterdir():
    if src_folder.is_dir():
        folder_name = src_folder.name

       
        if not (folder_name.isdigit() and len(folder_name) == 5):
            continue

        
        if folder_name not in folder_mapping:
            print(f"未找到映射关系的文件夹 {folder_name}")
            continue

        category = folder_mapping[folder_name]

        
        for img_file in src_folder.glob("*.jpg"):
            
            modality = img_file.stem  

            
            new_filename = f"{folder_name}{modality}.jpg"

            
            dest_dir = dst_root / category / modality
            dest_dir.mkdir(parents=True, exist_ok=True)

           
            shutil.copy(img_file, dest_dir / new_filename)
            print(f"已复制：{img_file} -> {dest_dir / new_filename}")

print("sucessful!")
