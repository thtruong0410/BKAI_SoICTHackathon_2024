import os
import json
import cv2

# Chuyển đổi nhãn YOLO sang COCO format
def convert_yolo_to_coco(yolo_label_file, img_width, img_height):
    coco_annotations = []
    with open(yolo_label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            category_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_center, y_center = x_center * img_width, y_center * img_height
            width, height = width * img_width, height * img_height
            x_min = x_center - width / 2
            y_min = y_center - height / 2

            annotation = {
                "bbox": [x_min, y_min, width, height],
                "category_id": int(category_id),
                "area": width * height,
                "iscrowd": 0
            }
            coco_annotations.append(annotation)
    return coco_annotations

# Tạo COCO JSON từ YOLO dataset
def convert_to_coco_format(image_dir, label_dir, output_file):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Định nghĩa danh mục (classes)
    categories = ["motor", "car", "bus", "container"]  # Sửa lại theo danh sách class của bạn
    for i, cat_name in enumerate(categories):
        coco_data["categories"].append({
            "id": i,
            "name": cat_name,
        })

    annotation_id = 1

    for image_file in os.listdir(image_dir):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_id = len(coco_data["images"]) + 1
            image_path = os.path.join(image_dir, image_file)
            img = cv2.imread(image_path)
            img_height, img_width, _ = img.shape

            # Thêm thông tin hình ảnh vào COCO dataset
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": img_width,
                "height": img_height
            })

            # Lấy đường dẫn file nhãn YOLO tương ứng
            yolo_label_file = os.path.join(label_dir, image_file.replace(".jpg", ".txt").replace(".png", ".txt"))
            if os.path.exists(yolo_label_file):
                annotations = convert_yolo_to_coco(yolo_label_file, img_width, img_height)
                for annotation in annotations:
                    annotation["image_id"] = image_id
                    annotation["id"] = annotation_id
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1

    # Ghi dữ liệu COCO vào tệp JSON
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=4)

# Đường dẫn thư mục
train_image_dir = r"/workspace/report_final/data/data_default/images/train"
train_label_dir = r"/workspace/report_final/data/data_default/labels/train"
test_image_dir = r"/workspace/report_final/data/data_default/images/test"
test_label_dir = r"/workspace/report_final/data/data_default/labels/test"

# Chuyển đổi YOLO train và test sang COCO format
convert_to_coco_format(train_image_dir, train_label_dir, r"/workspace/report_final/data/data_default/annotations/train_coco.json")
convert_to_coco_format(test_image_dir, test_label_dir, r"/workspace/report_final/data/data_default/annotations/test_coco.json")
