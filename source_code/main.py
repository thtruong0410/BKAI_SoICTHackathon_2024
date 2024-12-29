import os
import sys

# Sử dụng đường dẫn tuyệt đối với thư mục gốc của dự án
BASE_DIR = "/workspace/"
sys.path.append(os.path.abspath(BASE_DIR))

from tqdm import tqdm
from predict_yolo import predict_yolo
from Co_DETR.predict import predict_on_image, load_model
from ultralytics import YOLO
from sahi_infer import sahi
from wbf import process_wbf
from post_process import post_process
from filter_by_confidence import filter_confidence

# Sử dụng os.path.join để tạo đường dẫn, đảm bảo tính di động giữa các hệ điều hành
weights_list_yolo = [
    os.path.join(BASE_DIR, "checkpoint", "best_v8x_134.pt"),
    os.path.join(BASE_DIR, "checkpoint", "yolo11x_epoch10.pt")
]

# Làm tương tự với các đường dẫn checkpoint và config của Co_DETR
codetr_list = {
    os.path.join(BASE_DIR, "checkpoint", "epoch_6.pth"): os.path.join(BASE_DIR, "Co_DETR", "blackbox-cfg", "co_dino_5scale_swin_large_16e_o365tococo_1333.py"),
    os.path.join(BASE_DIR, "checkpoint", "epoch_2.pth"): os.path.join(BASE_DIR, "Co_DETR", "blackbox-cfg", "co_dino_5scale_swin_large_16e_o365tococo_960.py"),
    os.path.join(BASE_DIR, "checkpoint", "epoch_13.pth"): os.path.join(BASE_DIR, "Co_DETR", "blackbox-cfg", "co_dino_5scale_swin_large_16e_o365tococo_1333.py"),
    os.path.join(BASE_DIR, "checkpoint", "epoch_9.pth"): os.path.join(BASE_DIR, "Co_DETR", "blackbox-cfg", "co_dino_5scale_swin_large_16e_o365tococo_1333.py")
}

# Sử dụng os.path.join để tạo các đường dẫn
data_path = os.path.join(BASE_DIR, "data", "private_test")
output_path = os.path.join(BASE_DIR, "result")
os.makedirs(output_path, exist_ok=True)

def main():
    # Step 1: Predict Yolo
    for weights in weights_list_yolo:
        model = YOLO(weights)
        weight_name_yolo = os.path.splitext(os.path.basename(weights))[0]
        print(f"Processing YOLO model: {weight_name_yolo}")
        
        output_file_predict_yolo = os.path.join(output_path, f"predict_{weight_name_yolo}.txt")

        # Điều chỉnh confidence và image size dựa trên tên weights
        if weight_name_yolo == "best_v8x_134":
            conf = 0.0001
            imgsz = 1280
        else:
            conf = 0.01
            imgsz = 640
        
        # Uncomment dòng dưới đây khi bạn muốn thực thi predict_yolo
        predict_yolo(model, data_path, output_file_predict_yolo, conf, imgsz)

    # Step 2: Predict Codetr
    for checkpoint_path, config_path in codetr_list.items():
        print(f"Processing Co-DETR model:")
        print(f"Config: {config_path}")
        print(f"Checkpoint: {checkpoint_path}")

        # Load model
        model = load_model(config_path, checkpoint_path)
        file_name = os.path.basename(checkpoint_path)
        weight_name = file_name.split('.')[0]
        imgsz=""
        if "1333" in config_path:
            imgsz="1333"
        if "960" in config_path:
            imgsz="960"
        # Predict for each image
        for file in tqdm(os.listdir(data_path)):
            image_path = os.path.join(data_path, file)
            predict_on_image(model, image_path, file, os.path.join(output_path, f"predict_{weight_name}_{imgsz}.txt"))

    # Step 3: Weighted Box Fusion (WBF)
    # Ensure the "no_area" directory exists
    os.makedirs(os.path.join(output_path, "no_area"), exist_ok=True)

    wbf_output_file = os.path.join(output_path, "no_area", "predict_fused.txt")
    process_wbf(output_path, wbf_output_file)

    # Step 4: Post processing
    final_output_file = os.path.join(output_path, "no_area", "predict.txt")
    output_path_area = os.path.join(output_path, "no_area", "predict_delete_area_1280.txt")
    output_path_ratio = os.path.join(output_path, "no_area", "predict_delete_area_whratio_1280.txt")
    
    post_process(wbf_output_file, data_path, output_path_area, output_path_ratio, final_output_file)

    # Step 5: Filter confidence score
    output_filter = os.path.join(output_path, "no_area")
    filter_confidence(final_output_file, output_filter)

    # Step 6: Polygon
    # (Add polygon processing if needed)
    print("Object detection pipeline completed successfully!")

if __name__ == "__main__":
    main()