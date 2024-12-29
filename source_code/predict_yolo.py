from ultralytics import YOLO
import os
from tqdm import tqdm
import cv2

def xyxy_to_xywhn(box, width, height):
    x1, y1, x2, y2 = box
    # Convert absolute coordinates to normalized
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    x = (x1 / width) + (w / 2)
    y = (y1 / height) + (h / 2)
    return [x, y, w, h]
def xywhn_to_xyxy(box, width, height):
    x, y, w, h = box
    # Convert normalized coordinates to absolute
    x1 = (x - w/2) * width
    y1 = (y - h/2) * height
    x2 = (x + w/2) * width
    y2 = (y + h/2) * height
    return [x1, y1, x2, y2]

def show_results(img, filtered_detections):
    for det in filtered_detections:
        class_id = det[0]
        xywhn = det[1]
        conf = det[2]
        x1, y1, x2, y2 = xywhn_to_xyxy(xywhn, img.shape[1], img.shape[0])
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{class_id}: {float(conf):.2f}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Results", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict_yolo(model, data_path, output_path, confi, imgsz):
    with open(output_path, 'w') as f:
        for file in tqdm(os.listdir(data_path)):
            img_path = os.path.join(data_path, file)

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            # Predict 
            results = model(img, verbose=False, conf=float(confi), imgsz=imgsz)  # Lưu ý tham số conf
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = box.cls
                    xywh = box.xywhn[0]
                    conf = box.conf

                    # Nếu train model 7 class thì dùng cái này
                    if int(class_id) == 4:
                        class_id = 0
                    if int(class_id) == 5:
                        class_id = 1
                    if int(class_id) == 6:
                        class_id = 2
                    if int(class_id) == 7:
                        class_id = 3
                    ##########################################
                    f.write(f"{file} {int(class_id)} {round(float(xywh[0]), 4)} {round(float(xywh[1]), 4)} {round(float(xywh[2]), 4)} {round(float(xywh[3]), 4)} {round(float(conf), 7)}\n")

