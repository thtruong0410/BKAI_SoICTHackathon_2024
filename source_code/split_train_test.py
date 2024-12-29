import os
import shutil
import random
from tqdm import tqdm

# Đường dẫn dữ liệu
dataset_dir = '../data/train_20241023/daytime'
# dataset_dir = '../data/train_20241023/nighttime'
train_dir = '../data/dataset_2310/train'
test_dir = '../data/dataset_2310/test'

root_dir = "../data/dataset_2310"
yolo_dir = "../data/dataset_yolo"

# Đường dẫn YOLO
image_dirs = {
    "train": os.path.join(yolo_dir, "images", "train"),
    "test": os.path.join(yolo_dir, "images", "test"),
}
label_dirs = {
    "train": os.path.join(yolo_dir, "labels", "train"),
    "test": os.path.join(yolo_dir, "labels", "test"),
}

# Tỉ lệ chia dữ liệu
split_ratio = 0.9

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
for path in image_dirs.values():
    os.makedirs(path, exist_ok=True)
for path in label_dirs.values():
    os.makedirs(path, exist_ok=True)

error_files = []


def copy_files(files, src_dir, dst_dir):
    """Copy danh sách file từ src_dir sang dst_dir và chuyển đổi nhãn về 4 class."""
    for file_name in files:
        img_src = os.path.join(src_dir, file_name)
        txt_src = os.path.join(src_dir, file_name.replace('.jpg', '.txt'))
        
        # Kiểm tra tồn tại file
        if not os.path.exists(img_src):
            error_files.append(f"Lỗi: Không tìm thấy file ảnh {img_src}")
            continue
        if not os.path.exists(txt_src):
            error_files.append(f"Lỗi: Không tìm thấy file nhãn {txt_src}")
            continue
        
        # Copy file ảnh
        shutil.copy(img_src, os.path.join(dst_dir, file_name))
        
        # Chuyển đổi và copy file nhãn
        with open(txt_src, 'r') as src_label_file:
            lines = src_label_file.readlines()
        
        new_labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                error_files.append(f"Lỗi định dạng nhãn: {txt_src}")
                continue
            class_id = int(parts[0])
            # Chuyển đổi class ID
            if class_id == 4:
                class_id = 0
            elif class_id == 5:
                class_id = 1
            elif class_id == 6:
                class_id = 2
            elif class_id == 7:
                class_id = 3
            else:
                # Bỏ qua các class không nằm trong danh sách
                continue
            new_labels.append(f"{class_id} {' '.join(parts[1:])}")
        
        # Ghi file nhãn mới
        dst_label_path = os.path.join(dst_dir, file_name.replace('.jpg', '.txt'))
        with open(dst_label_path, 'w') as dst_label_file:
            dst_label_file.write("\n".join(new_labels))


def split_and_copy_data(data_dir, train_dir, test_dir, split_ratio):
    """Phân chia dữ liệu thành tập train/test và copy sang thư mục tương ứng."""
    data_by_cam = {}

    # Nhóm file theo camera ID
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.jpg'):
            cam_id = file_name.split('_')[1]
            data_by_cam.setdefault(cam_id, []).append(file_name)

    # Phân chia dữ liệu và copy
    for cam_id, files in data_by_cam.items():
        random.shuffle(files)
        split_index = int(len(files) * split_ratio)
        train_files = files[:split_index]
        test_files = files[split_index:]

        copy_files(train_files, data_dir, train_dir)
        copy_files(test_files, data_dir, test_dir)


def copy_yolo_files(src_dir, dest_image_dir, dest_label_dir):
    """Copy file ảnh và nhãn theo định dạng YOLO."""
    for file_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file_name)
        if file_name.endswith(".jpg"):
            shutil.copy(src_path, os.path.join(dest_image_dir, file_name))
        elif file_name.endswith(".txt"):
            shutil.copy(src_path, os.path.join(dest_label_dir, file_name))


# Phân chia dữ liệu thành train/test
split_and_copy_data(dataset_dir, train_dir, test_dir, split_ratio)

# Copy dữ liệu YOLO
copy_yolo_files(test_dir, image_dirs["test"], label_dirs["test"])
copy_yolo_files(train_dir, image_dirs["train"], label_dirs["train"])

# Hiển thị kết quả
if error_files:
    print("Các file lỗi không được copy:")
    for error in error_files:
        print(error)
else:
    print("Hoàn thành phân chia và copy dữ liệu mà không có lỗi.")
