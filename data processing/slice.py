import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import warnings

def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    return data, affine, img

def estimate_tumor_by_threshold(data, percentile=99):
    threshold = np.percentile(data, percentile)
    mask = data > threshold
    print(f"使用强度阈值 {threshold:.2f} (百分位数 {percentile}%) 估计肿瘤区域。")
    return mask

def find_max_cross_section(data, mask, slice_axis=2):
    if slice_axis >= data.ndim:
        raise ValueError(f" {slice_axis} 超出数据维度 {data.ndim}")
    areas = []
    for i in range(data.shape[slice_axis]):
        if slice_axis == 0:
            slice_mask = mask[i, :, :]
        elif slice_axis == 1:
            slice_mask = mask[:, i, :]
        else:  # axis=2
            slice_mask = mask[:, :, i]
        area = np.sum(slice_mask > 0)
        areas.append(area)

    best_idx = int(np.argmax(areas))
    best_area = areas[best_idx]
    return best_idx, best_area, areas

def extract_slice(data, slice_idx, slice_axis=2):
    if slice_axis == 0:
        slice_data = data[slice_idx, :, :]
    elif slice_axis == 1:
        slice_data = data[:, slice_idx, :]
    else:
        slice_data = data[:, :, slice_idx]
    return slice_data

def save_slice_as_png(slice_data, output_path, cmap='gray', normalize=True):
    if normalize:
        vmin, vmax = np.percentile(slice_data, (0.5, 99.5)) 
        slice_data = np.clip(slice_data, vmin, vmax)
        slice_data = (slice_data - vmin) / (vmax - vmin) * 255
        slice_data = slice_data.astype(np.uint8)
    plt.imsave(output_path, slice_data, cmap=cmap, format='png')

def save_slice_as_nifti(slice_data, affine, output_path, slice_axis=2, original_shape=None):
    
    if original_shape is None:
        if slice_axis == 0:
            vol_3d = slice_data[np.newaxis, :, :]
        elif slice_axis == 1:
            vol_3d = slice_data[:, np.newaxis, :]
        else:
            vol_3d = slice_data[:, :, np.newaxis]
    else:
        vol_3d = np.zeros(original_shape, dtype=slice_data.dtype)
        if slice_axis == 0:
            vol_3d[slice_idx, :, :] = slice_data
        elif slice_axis == 1:
            vol_3d[:, slice_idx, :] = slice_data
        else:
            vol_3d[:, :, slice_idx] = slice_data
    new_img = nib.Nifti1Image(vol_3d, affine)
    nib.save(new_img, output_path)

def main():
    parser = argparse.ArgumentParser(description='肿瘤最大横断面切片')
    parser.add_argument('--input', '-i', required=True, help='输入 (.nii.gz)')
    parser.add_argument('--mask', '-m', help=' (.nii.gz) 强度阈值估计')
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--slice-axis', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--threshold-percentile', type=float, default=99)
    parser.add_argument('--no-normalize', action='store_true')
    args = parser.parse_args()

    print(args.input)
    data, affine, img = load_nifti(args.input)
    print(f"数据形状: {data.shape}, 数据类型: {data.dtype}")
    
    if args.mask and Path(args.mask).exists():
        print(args.mask)
        mask_data, _, _ = load_nifti(args.mask)
        if mask_data.shape != data.shape:
            raise ValueError(f"掩膜形状 {mask_data.shape} 与原始图像形状 {data.shape} 不匹配")
        mask = mask_data > 0  # 二值化
    else:
        mask = estimate_tumor_by_threshold(data, args.threshold_percentile)

    best_idx, best_area, all_areas = find_max_cross_section(data, mask, args.slice_axis)
    print(f"索引: {best_idx} (肿瘤面积: {best_area} 像素)")
    print(f"切片范围: 0 ~ {data.shape[args.slice_axis]-1}")
    slice_data = extract_slice(data, best_idx, args.slice_axis)

    output_path = Path(args.output)
    if output_path.suffix.lower() == '.png':
        save_slice_as_png(slice_data, output_path, normalize=not args.no_normalize)
        print(f"切片保存 PNG: {output_path}")
    elif output_path.suffix.lower() in ('.nii', '.nii.gz'):
        save_slice_as_nifti(slice_data, affine, output_path, args.slice_axis, data.shape)
        print(f"保存 NIfTI: {output_path}")
    try:
        plt.figure(figsize=(6,6))
        plt.imshow(slice_data, cmap='gray')
        plt.title(f"切片 (索引 {best_idx}, 面积 {best_area})")
        plt.axis('off')
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()
