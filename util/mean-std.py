import os
import cv2
import numpy as np
from tqdm import tqdm


def collect_image_paths_from_lists(data_root, data_lists, deduplicate=True):
    """
    从多个 list 文件中收集图像路径
    每行格式例如: Images/xxx.png 3
    """
    all_img_paths = []
    seen = set()

    for data_list in data_lists:
        if not os.path.isfile(data_list):
            raise FileNotFoundError(f"data_list not found: {data_list}")

        with open(data_list, 'r', encoding='utf-8-sig') as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            img_rel_path = parts[0]
            img_path = os.path.join(data_root, img_rel_path)

            if deduplicate:
                if img_path not in seen:
                    seen.add(img_path)
                    all_img_paths.append(img_path)
            else:
                all_img_paths.append(img_path)

    return all_img_paths


def compute_mean_std_from_image_paths(image_paths):
    """
    统计 RGB 图像在 [0,1] 范围下的 mean/std
    """
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    valid_count = 0

    for img_path in tqdm(image_paths, desc="Computing mean/std"):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[Warning] failed to read: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # 转到 [0,1]

        h, w, c = img.shape
        assert c == 3, f"Image is not 3-channel RGB: {img_path}"

        img = img.reshape(-1, 3)  # [H*W, 3]

        pixel_sum += img.sum(axis=0)
        pixel_sq_sum += (img ** 2).sum(axis=0)
        pixel_count += img.shape[0]
        valid_count += 1

    if pixel_count == 0:
        raise RuntimeError("No valid image found.")

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    return mean.tolist(), std.tolist(), valid_count, pixel_count


if __name__ == "__main__":
    data_root = r"/home/pc/Data/LXW/MAPTNet-main/TYUST_FLA"
    train_list1 = r"/home/pc/Data/LXW/MAPTNet-main/TYUST_FLA_list/train/fold0_defect.txt"
    train_list2 = r"/home/pc/Data/LXW/MAPTNet-main/TYUST_FLA_list/train/fold1_defect.txt"
    train_list3 = r"/home/pc/Data/LXW/MAPTNet-main/TYUST_FLA_list/train/fold2_defect.txt"

    data_lists = [train_list1, train_list2, train_list3]

    # True: 三个 list 合并后去重统计（推荐）
    # False: 三个 list 合并后不去重，重复图像会重复计数
    deduplicate = True

    image_paths = collect_image_paths_from_lists(
        data_root=data_root,
        data_lists=data_lists,
        deduplicate=deduplicate
    )

    print(f"Total image paths collected: {len(image_paths)}")
    print(f"Deduplicate: {deduplicate}")

    mean, std, valid_count, pixel_count = compute_mean_std_from_image_paths(image_paths)

    print("\n==== [0,1] scale ====")
    print(f"valid images = {valid_count}")
    print(f"total pixels = {pixel_count}")
    print("mean =", [round(x, 6) for x in mean])
    print("std  =", [round(x, 6) for x in std])

    print("\n==== [0,255] scale ====")
    mean_255 = [x * 255.0 for x in mean]
    std_255 = [x * 255.0 for x in std]
    print("mean =", [round(x, 6) for x in mean_255])
    print("std  =", [round(x, 6) for x in std_255])