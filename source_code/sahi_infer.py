from ultralytics import YOLO
import os
from tqdm import tqdm
import cv2
import torch
from torchvision.ops import box_iou
import numpy as np

def detect_full_image(model, img_path, conf_threshold=0.005):
    results = model(img_path, verbose=False, conf=conf_threshold)
    list_full = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = box.cls.cpu().numpy()[0]
            xywhn = box.xywhn[0].cpu().numpy()
            conf = box.conf.cpu().numpy()[0]
            
            if int(class_id) == 4:
                class_id = 0
            elif int(class_id) == 5:
                class_id = 1
            elif int(class_id) == 6:
                class_id = 2
            elif int(class_id) == 7:
                class_id = 3
                
            list_full.append([class_id, xywhn, conf])
    
    return list_full

def detect_sliding_windows(model, img_path, stride=512, conf_threshold=0.005):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    list_crop = []
    
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            window = img[i:i+stride, j:j+stride]
            if window.size == 0:
                continue
                
            results = model(window, verbose=False, conf=conf_threshold)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = box.cls.cpu().numpy()[0]
                    xywh = box.xywh[0].cpu().numpy()
                    conf = box.conf.cpu().numpy()[0]
                    
                    adjusted_x = (xywh[0] + j) / w
                    adjusted_y = (xywh[1] + i) / h
                    adjusted_w = xywh[2] / w
                    adjusted_h = xywh[3] / h
                    adjusted_xywhn = [adjusted_x, adjusted_y, adjusted_w, adjusted_h]
                    
                    if int(class_id) == 4:
                        class_id = 0
                    elif int(class_id) == 5:
                        class_id = 1
                    elif int(class_id) == 6:
                        class_id = 2
                    elif int(class_id) == 7:
                        class_id = 3
                    
                    list_crop.append([class_id, adjusted_xywhn, conf])
    
    return list_crop

def xywhn_to_xyxy(box, width, height):
    x, y, w, h = box
    x1 = (x - w/2) * width
    y1 = (y - h/2) * height
    x2 = (x + w/2) * width
    y2 = (y + h/2) * height
    return [x1, y1, x2, y2]

def filter_crop_detections(list_full, list_crop, width, height, iou_threshold=0.3):
    if not list_full or not list_crop:
        return list_full + list_crop
    
    # Convert all boxes to xyxy format for IoU calculation
    full_boxes = torch.tensor([xywhn_to_xyxy(det[1], width, height) for det in list_full], dtype=torch.float32)
    crop_boxes = torch.tensor([xywhn_to_xyxy(det[1], width, height) for det in list_crop], dtype=torch.float32)
    
    # Calculate IoU between all pairs of boxes
    iou_matrix = box_iou(crop_boxes, full_boxes)
    
    # Find maximum IoU for each crop box
    max_ious, _ = torch.max(iou_matrix, dim=1)
    
    # Filter crop detections based on IoU threshold
    filtered_crop = []
    for idx, (max_iou, crop_det) in enumerate(zip(max_ious, list_crop)):
        if max_iou < iou_threshold:
            filtered_crop.append(crop_det)
    
    # Combine full detections with filtered crop detections
    return list_full + filtered_crop

def show_results(img, filtered_detections):
    img_copy = img.copy()
    for det in filtered_detections:
        class_id = det[0]
        xywhn = det[1]
        conf = det[2]
        x1, y1, x2, y2 = xywhn_to_xyxy(xywhn, img.shape[1], img.shape[0])
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img_copy, f"{class_id}: {conf:.2f}", (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Results", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sahi(model, data_path, output_path):
    conf_threshold = 0.01
    iou_threshold = 0.3 
    stride = 768
    
    with open(output_path, 'w') as f:
        for file in tqdm(os.listdir(data_path)):
            img_path = os.path.join(data_path, file)
            img = cv2.imread(img_path)
            try:
                height, width = img.shape[:2]
                list_full = detect_full_image(model, img_path, conf_threshold)
                list_crop = detect_sliding_windows(model, img_path, stride, conf_threshold)
                filtered_detections = filter_crop_detections(list_full, list_crop, width, height, iou_threshold)
            except Exception as e:
                print(f"Không thể xử lý file {file}: {e}")
                continue
            for class_id, xywhn, conf in filtered_detections:
                f.write(f"{file} {int(class_id)} {round(float(xywhn[0]), 4)} {round(float(xywhn[1]), 4)} "
                    f"{round(float(xywhn[2]), 4)} {round(float(xywhn[3]), 4)} {round(float(conf), 7)}\n")

# if __name__ == "__main__":
#     main()