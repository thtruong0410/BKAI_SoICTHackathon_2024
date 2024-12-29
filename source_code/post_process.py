import os
import cv2
import numpy as np
from tqdm import tqdm


def check_box_ratio(w, h, img, threshold=5):
    """
    Kiểm tra tỉ lệ giữa chiều rộng và chiều cao của box so với ngưỡng.
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = w * img_w, h * img_h
    ratio = max(w / h, h / w)
    return ratio <= threshold, ratio


def filter_boxes_by_ratio(result_path, img_dir, output_path, threshold=5):
    """
    Lọc các box trong file kết quả dựa trên tỉ lệ w/h và lưu lại kết quả.
    """
    # Đọc file kết quả
    result_cam = {}
    with open(result_path, 'r') as f:
        for line in f:
            img_name, class_id, cxn, cyn, wn, hn, conf = line.strip().split(' ')
            result_cam.setdefault(img_name, []).append([class_id, cxn, cyn, wn, hn, conf])

    filtered_results = []
    boxes_removed_ratio = 0

    for img_name in tqdm(result_cam):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Không thể đọc ảnh: {img_name}")
            continue

        for box in result_cam[img_name]:
            class_id, cxn, cyn, wn, hn, conf = box
            check_wh, ratio = check_box_ratio(float(wn), float(hn), img, threshold)

            if check_wh:
                filtered_results.append(f"{img_name} {' '.join(box)}\n")
            else:
                boxes_removed_ratio += 1

    # Lưu kết quả lọc
    with open(output_path, 'w') as f:
        f.writelines(filtered_results)

    # Thống kê
    total_boxes = sum(len(boxes) for boxes in result_cam.values())
    print(f"Đã lưu kết quả lọc vào {output_path}")
    print(f"Tổng số ảnh: {len(result_cam)}")
    print(f"Số lượng box trước khi lọc: {total_boxes}")
    print(f"Số lượng box sau khi lọc: {len(filtered_results)}")
    print(f"Số lượng box bị loại do tỉ lệ w/h vượt ngưỡng {threshold}: {boxes_removed_ratio}")


def filter_boxes_by_area(input_path, img_dir, output_path, min_area=250):
    """
    Lọc các box có diện tích nhỏ hơn ngưỡng và lưu lại kết quả.
    """
    with open(input_path, "r") as f:
        lines = f.readlines()

    filtered_lines = []
    boxes_removed_area = 0

    for line in tqdm(lines):
        img_name, class_id, cxn, cyn, wn, hn, conf = line.strip().split(" ")
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Không thể đọc ảnh: {img_name}")
            continue

        im_h, im_w = img.shape[:2]
        cxn, cyn, wn, hn = float(cxn), float(cyn), float(wn), float(hn)

        # Tính tọa độ bounding box
        x1 = max(0, int(cxn * im_w - (wn * im_w) / 2))
        y1 = max(0, int(cyn * im_h - (hn * im_h) / 2))
        x2 = min(im_w, int(cxn * im_w + (wn * im_w) / 2))
        y2 = min(im_h, int(cyn * im_h + (hn * im_h) / 2))

        # Diện tích box
        area_box = (x2 - x1) * (y2 - y1)
        if area_box >= min_area:
            filtered_lines.append(line)
        else:
            boxes_removed_area += 1

    # Lưu kết quả
    with open(output_path, "w") as f:
        f.writelines(filtered_lines)

    # Thống kê
    print(f"Đã lưu kết quả lọc vào {output_path}")
    print(f"Số lượng box trước khi lọc: {len(lines)}")
    print(f"Số lượng box sau khi lọc: {len(filtered_lines)}")
    print(f"Số lượng box bị loại do diện tích < {min_area}: {boxes_removed_area}")


# # Đường dẫn dữ liệu
# result_path = r"E:\BKAI\Code\Report\results\predict_1280\predict.txt"
# output_path_ratio = r"E:\BKAI\Code\Report\results\predict_1280\predict_delete_area_wh_ratio_1280.txt"
# output_path_area = r"E:\BKAI\Code\Report\results\predict_1280\predict_delete_area_1280.txt"
# public_test_path = r"E:\BKAI\public test"

def post_process(wbf_output_file, data_path, output_path_area, output_path_ratio, final_output_file):

    os.makedirs(os.path.dirname(output_path_area), exist_ok=True)
    os.makedirs(os.path.dirname(output_path_ratio), exist_ok=True)

    # Lọc box theo tỷ lệ w/h
    filter_boxes_by_ratio(wbf_output_file, data_path, output_path_ratio, threshold=5)

    # Lưu kết quả cuối cùng
    print("Hoàn tất lọc box. Sao chép kết quả cuối cùng...")
    os.rename(output_path_ratio, final_output_file)
    print(f"Kết quả cuối cùng đã được lưu tại: {final_output_file}")
    