from shapely.geometry import Polygon, box

# Define the polygons
polygons = [
    [(307, 453), (370, 466), (367, 531), (336, 549), (302, 541), (287, 468)],
    [(840, 562), (986, 560), (1005, 695), (870, 705)]
]

# Create Polygon objects from the list of coordinates
polygons = [Polygon(p) for p in polygons]

# Function to convert YOLO format to pixel coordinates
def yolo_to_pixels(image_width, image_height, x_center, y_center, width, height):
    x_min = (x_center - width / 2) * image_width
    y_min = (y_center - height / 2) * image_height
    x_max = (x_center + width / 2) * image_width
    y_max = (y_center + height / 2) * image_height
    return x_min, y_min, x_max, y_max

# Function to calculate intersection area ratio
def get_intersection_area_ratio(polygon, x_min, y_min, x_max, y_max):
    # Create a box from the bounding box coordinates
    bounding_box = box(x_min, y_min, x_max, y_max)
    # Calculate intersection area with the polygon
    intersection = polygon.intersection(bounding_box)
    intersection_area = intersection.area
    box_area = bounding_box.area
    return intersection_area / box_area

# Read the input file
input_file = r'D:\BKAI\Traffic Vehicle Detection-20241022T055854Z-001\predict_7572.txt'  # Replace with your actual file path
output_file = r'D:\BKAI\Traffic Vehicle Detection-20241022T055854Z-001\predict_7572_poligon.txt'  # File to save valid boxes

image_width = 1280
image_height = 720

valid_boxes = []
count = 0
# Read the input file and process each line
with open(input_file, 'r') as f:
    for line in f:
        if "cam_11" not in line:
            valid_boxes.append(line.strip())
            continue
        parts = line.strip().split()  # Split by whitespace
        filename = parts[0]
        x_center, y_center, width, height = map(float, parts[2:6])  # Extract the bounding box coordinates
        
        # Convert YOLO coordinates to pixel coordinates
        x_min, y_min, x_max, y_max = yolo_to_pixels(image_width, image_height, x_center, y_center, width, height)
        
        # Check if the box intersects with any polygon
        keep_box = True
        for polygon in polygons:
            intersection_ratio = get_intersection_area_ratio(polygon, x_min, y_min, x_max, y_max)
            
            # If the intersection area is greater than 90% of the bounding box, discard it
            if intersection_ratio >= 0.9:
                keep_box = False
                count+=1
                break
        
        # If box is valid, add to the list
        if keep_box:
            valid_boxes.append(line.strip())  # Keep the original line (without modification)

# Write valid boxes to the output file
with open(output_file, 'w') as f:
    for box in valid_boxes:
        f.write(box + '\n')
print(count)
print(f"Valid boxes saved to {output_file}")
