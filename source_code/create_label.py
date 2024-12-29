import os
import cv2
import numpy as np
from ultralytics import YOLO  # Thư viện YOLOv8

# Hàm đọc các bounding box từ file YOLO
def read_yolo_boxes(label_path):
    with open(label_path, 'r') as file:
        boxes = []
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id, x_center, y_center, w, h = map(float, parts[:5])
                boxes.append((class_id, x_center, y_center, w, h))
    return boxes

# Hàm tính toán IoU giữa 2 bounding box
def calculate_iou(box1, box2):
    _, x1, y1, w1, h1 = box1
    _, x2, y2, w2, h2 = box2

    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

    inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = inter_w * inter_h
    area1, area2 = w1 * h1, w2 * h2
    union = area1 + area2 - intersection
    return intersection / union if union else 0

# Chạy YOLOv8 trên thư mục ảnh và thêm box mới vào labels_new
def detect_additional_objects(images_folder, labels_folder, labels_new_folder, model_path, iou_threshold=0.5):
    model = YOLO(model_path)  # Load mô hình YOLOv8
    
    # Tạo thư mục labels_new nếu chưa có
    os.makedirs(labels_new_folder, exist_ok=True)
    
    for img_file in os.listdir(images_folder):
        if not img_file.endswith(('.jpg', '.png')):
            continue
        
        img_path = os.path.join(images_folder, img_file)
        label_path = os.path.join(labels_folder, os.path.splitext(img_file)[0] + '.txt')
        label_new_path = os.path.join(labels_new_folder, os.path.splitext(img_file)[0] + '.txt')
        
        existing_boxes = read_yolo_boxes(label_path) if os.path.exists(label_path) else []
        new_boxes = []
        
        # Chạy YOLOv8 để detect đối tượng mới
        results = model(img_path)
        
        for detection in results[0].boxes:
            x_center, y_center, w, h = detection.xywhn[0]  # YOLOv8 Normalized bbox (x, y, w, h)
            class_id = detection.cls[0].item()
            box_new = (class_id, x_center, y_center, w, h)
            
            # Kiểm tra box mới có trùng với box cũ không
            if all(calculate_iou(box_new, box_existing) < iou_threshold for box_existing in existing_boxes):
                new_boxes.append(box_new)
        
        # Ghi cả box cũ và box mới vào file label mới
        with open(label_new_path, 'w') as file:
            for box in existing_boxes + new_boxes:
                file.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
        if len(new_boxes) != 0:
            print(f"Processed {img_file}: Added {len(new_boxes)} new boxes to {label_new_path}")

# Ví dụ sử dụng:
images_folder = r'..\data\Merge\images\train'
labels_folder = r'..\data\Merge\labels\train'
labels_new_folder = r'..\data\Merge\labels_new'
model_path = r'..\checkpoint\best_v8l_4class.pt'  # Đường dẫn tới mô hình YOLOv8

detect_additional_objects(images_folder, labels_folder, labels_new_folder, model_path)
