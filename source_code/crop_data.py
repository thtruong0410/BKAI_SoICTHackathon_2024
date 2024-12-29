import os
import cv2
import random
from tqdm import tqdm

data_path = "../data/dataset_yolo"
images_path = os.path.join(data_path, "images", "train")
labels_path = os.path.join(data_path, "labels", "train")
def random_crop(image, labels, crop_size=(700, 700), wh_ratio_threshold=10):
    h, w, _ = image.shape
    crop_w, crop_h = crop_size
    x_start = random.randint(0, w - crop_w)
    y_start = random.randint(0, h - crop_h)
    cropped_image = image[y_start:y_start + crop_h, x_start:x_start + crop_w]
    new_labels = []
    for label in labels:
        if len(label) == 5:
            class_id, x_center, y_center, width, height = label
            conf = None
        else:
            class_id, x_center, y_center, width, height, conf = label
        x = x_center * w
        y = y_center * h
        box_w = width * w
        box_h = height * h
        x_min = x - box_w / 2
        x_max = x + box_w / 2
        y_min = y - box_h / 2
        y_max = y + box_h / 2
        x_min_cropped = max(x_min, x_start)
        x_max_cropped = min(x_max, x_start + crop_w)
        y_min_cropped = max(y_min, y_start)
        y_max_cropped = min(y_max, y_start + crop_h)
        if x_min_cropped >= x_max_cropped or y_min_cropped >= y_max_cropped:
            continue
        new_box_w = x_max_cropped - x_min_cropped
        new_box_h = y_max_cropped - y_min_cropped
        if new_box_w <= 0 or new_box_h <= 0 or max(new_box_w / new_box_h, new_box_h / new_box_w) > wh_ratio_threshold:
            continue
        new_x_center = (x_min_cropped + new_box_w / 2 - x_start) / crop_w
        new_y_center = (y_min_cropped + new_box_h / 2 - y_start) / crop_h
        new_width = new_box_w / crop_w
        new_height = new_box_h / crop_h
        if conf is None:
            new_labels.append((int(class_id), new_x_center, new_y_center, new_width, new_height))
        else:
            new_labels.append((int(class_id), new_x_center, new_y_center, new_width, new_height, conf))

    return cropped_image, new_labels

def save_cropped_data(image, labels, image_name, label_name, count):
    new_path = "../data/data_crop"
    os.makedirs(new_path, exist_ok=True)
    os.makedirs(os.path.join(new_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(new_path, "labels", "train"), exist_ok=True)
    new_image_path = os.path.join(new_path, "images", "train", f"{image_name}_{count}.jpg")
    new_label_path = os.path.join(new_path, "labels", "train", f"{label_name}_{count}.txt")
    cv2.imwrite(new_image_path, image)

    with open(new_label_path, 'w') as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")
count = 0
for image_file in tqdm(os.listdir(images_path), desc="Processing images"):
    if image_file.endswith(".jpg"):
        try:
            image_path = os.path.join(images_path, image_file)
            image_name = os.path.splitext(image_file)[0]
            label_path = os.path.join(labels_path, f"{image_name}.txt")

            image = cv2.imread(image_path)
            labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    labels.append(tuple(map(float, line.strip().split())))
            cropped_image, new_labels = random_crop(image, labels)

            save_cropped_data(cropped_image, new_labels, image_name, image_name, count)
            count += 1
        except Exception as e:
            print(f"Error: {e}")
            continue
