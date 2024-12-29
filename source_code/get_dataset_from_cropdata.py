import os
import random
import shutil
from tqdm import tqdm

def select_images_and_labels(image_dir, label_dir, output_dir, quantity=5000):
    # List of image and label files
    image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    label_list = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    # List of required images from image_01 to image_09
    required_images = [f"image_{str(i).zfill(2)}.jpg" for i in range(1, 10)]
    required_labels = [f"image_{str(i).zfill(2)}.txt" for i in range(1, 10)]

    # Create output directories for selected images and labels
    output_image_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Lists to store selected images and labels
    all_selected_images = []
    all_selected_labels = []
    
    for image in tqdm(required_images, desc="Selecting images..."):
        # Get related labels for this image
        prefix = image.split('.')[0]  # Get the image name part (e.g., "image_01")
        related_labels = [f for f in label_list if f.startswith(prefix)]

        if image in image_list:
            # Copy the image and its related labels to the output directories
            shutil.copy(os.path.join(image_dir, image), os.path.join(output_image_dir, image))
            for label in related_labels:
                shutil.copy(os.path.join(label_dir, label), os.path.join(output_label_dir, label))

            # Add to the list of selected images and labels
            all_selected_images.append(image)
            all_selected_labels.extend(related_labels)

    # List of remaining images and labels after selecting image_01 to image_09
    remaining_images = [f for f in image_list if f not in required_images]
    remaining_labels = [f for f in label_list if f not in required_labels]

    # Calculate how many more images are needed
    remaining_images_to_select = quantity - len(all_selected_images)
    if remaining_images_to_select > 0:
        # Randomly select additional images
        random_images = random.sample(remaining_images, remaining_images_to_select)
        random_labels = [f.replace('.jpg', '.txt') for f in random_images]

        # Copy the selected random images and labels to the output directories
        for img, lbl in zip(random_images, random_labels):
            shutil.copy(os.path.join(image_dir, img), os.path.join(output_image_dir, img))
            shutil.copy(os.path.join(label_dir, lbl), os.path.join(output_label_dir, lbl))

            all_selected_images.append(img)
            all_selected_labels.append(lbl)

    print(f"Selected {len(all_selected_images)} images and labels, including images from image_01 to image_09.")

# Usage
image_dir = r"../data/data_crop/images/train"
label_dir = r"../data/data_crop/labels/train"
output_dir = r"../data/data_crop_selected"
select_images_and_labels(image_dir, label_dir, output_dir, quantity=6000)
