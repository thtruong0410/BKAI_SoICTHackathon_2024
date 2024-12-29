from Co-DETR.mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
import argparse
import os
from  tqdm import tqdm

def load_model(config_path, checkpoint_path):
    # Initialize the detector
    model = init_detector(config_path, checkpoint_path, device='cuda:0')
    return model

def predict_on_image(model, image_path, file, results_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")

    # Run inference on the image
    results = inference_detector(model, img)
    im_h, im_w, _ = img.shape
    # Process results to extract bounding boxes, labels, and scores
    bbox_result, _ = results, None
    bboxes = np.vstack(bbox_result)
    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
    labels = np.concatenate(labels)
    
    score_thr = 0.05
    if score_thr > 0:
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        scores = scores[inds]
    
    # Draw bounding boxes and labels on the image
    for label, score, bbox in zip(labels, scores, bboxes):
        bbox = list(map(int, bbox))
        class_id = int(label)  # Assuming class IDs start at 1
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cxn, cyn, wn, hn = cx / im_w, cy / im_h, w / im_w, h / im_h

        conf = score
        with open(results_path, "a") as f:
            f.write(f"{file} {int(class_id)} {cxn} {cyn} {wn} {hn} {conf}\n")

# def predict_codetr(config_codetr, weight_codetr, data_path, output_file_predict_codetr):
#     model = load_moel(config_codetr, weight_codetr)
#     for file in tqdm(os.listdir(data_path)):
#         image_path = os.path.join(data_path, file)
#         idx = weight.split(".")[0]
#         predict_on_image(model, image_path, file, output_file_predict_codetr)

# weight_path = r"/kaggle/input/checkpoint-codetr"

# results_dir = r'/kaggle/working/results'
# os.makedirs(results_dir, exist_ok=True)

# public_test_path = r"/kaggle/input/public-test/public test"
# # checkpoint_path = "E:/BKAI/weights/codetr_epoch1.pth"
# config_path = r"/kaggle/working/Co-DETR/blackbox-cfg/co_dino_5scale_swin_large_16e_o365tococo.py"

# def predict_codetr(config_path, weight, results_dir)
#     model = load_model(config_path, weight)
#     for file in tqdm(os.listdir(private_test_path)):
#         image_path = os.path.join(private_test_path, file)
#         idx = weight.split(".")[0]
#         results_path = os.path.join(results_dir, f"predict_{idx}.txt")
#         # print(results_path)
#         predict_on_image(model, image_path, file, results_path)