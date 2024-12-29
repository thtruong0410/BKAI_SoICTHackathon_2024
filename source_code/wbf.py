import os
from ensemble_boxes import weighted_boxes_fusion


def read_predictions(file_path):
    """
    Đọc dự đoán từ tệp và trả về dưới dạng từ điển.
    """
    prediction = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_name = parts[0]
            cls = int(parts[1])
            x_center, y_center, width, height, conf = map(float, parts[2:])
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            if file_name not in prediction:
                prediction[file_name] = {"boxes": [], "scores": [], "labels": []}
            prediction[file_name]["boxes"].append([x_min, y_min, x_max, y_max])
            prediction[file_name]["scores"].append(conf)
            prediction[file_name]["labels"].append(cls)
    return prediction


def save_fused_predictions(predictions, output_path):
    """
    Lưu kết quả dự đoán sau khi áp dụng WBF vào tệp.
    """
    with open(output_path, 'w') as f:
        for file_name, data in predictions.items():
            boxes, scores, labels = data['boxes'], data['scores'], data['labels']
            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min, x_max, y_max = box
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                f.write(f"{file_name} {int(label)} {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f} {score:.6f}\n")


def apply_wbf(predictions_list, weights, iou_thr=0.7, skip_box_thr=0.0001):
    """
    Áp dụng Weighted Boxes Fusion (WBF) trên danh sách các dự đoán.
    """
    fused_predictions = {}
    all_file_names = set()  # Lấy tập hợp tất cả các file_name từ tất cả các dự đoán
    for predictions in predictions_list:
        all_file_names.update(predictions.keys())

    for file_name in all_file_names:
        # Thu thập dữ liệu của từng tệp dự đoán (nếu tồn tại)
        boxes_list = []
        scores_list = []
        labels_list = []

        for pred in predictions_list:
            if file_name in pred:  # Chỉ thêm nếu file_name có trong dự đoán
                boxes_list.append(pred[file_name]["boxes"])
                scores_list.append(pred[file_name]["scores"])
                labels_list.append(pred[file_name]["labels"])
            else:
                # Nếu không có file_name trong một dự đoán, thêm danh sách rỗng
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])

        # Áp dụng WBF (chỉ khi có ít nhất một mô hình có dự đoán hợp lệ)
        if any(scores_list):  # Kiểm tra xem có bất kỳ `scores_list` nào không rỗng
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list, weights=weights,
                iou_thr=iou_thr, skip_box_thr=skip_box_thr
            )

            # Lưu kết quả sau khi áp dụng WBF
            fused_predictions[file_name] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            }
    return fused_predictions


def process_wbf(results_dir, output_file, iou_thr=0.5, skip_box_thr=0.0001):
    """
    Quét tất cả các tệp dự đoán trong thư mục `results_dir`, áp dụng WBF và lưu kết quả vào `output_file`.
    """
    prediction_files = [
        os.path.join(results_dir, file)
        for file in os.listdir(results_dir) if file.endswith(".txt")
    ]

    # Đọc dự đoán từ tất cả các tệp
    predictions_list = [read_predictions(file) for file in prediction_files]

    # Khởi tạo trọng số bằng nhau cho tất cả các mô hình
    weights = [1] * len(prediction_files)

    # Áp dụng WBF
    fused_predictions = apply_wbf(predictions_list, weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    # Lưu kết quả đã gộp
    save_fused_predictions(fused_predictions, output_file)
    print(f"Kết quả sau khi áp dụng WBF đã được lưu tại: {output_file}")
