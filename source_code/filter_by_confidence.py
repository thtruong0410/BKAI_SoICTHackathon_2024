import os

def filter_confidence(txt_path, output_path):
    base_name = os.path.basename(txt_path)
    file_name, ext = os.path.splitext(base_name)
    output_txt_path = os.path.join(output_path, f"{file_name}_filtered{ext}")
# Ngưỡng confidence score theo từng class
    thresholds = {
        0: 0.14,  # Class 0
        1: 0.21,  # Class 1
        2: 0.305,  # Class 2
        3: 0.21  # Class 3
    }

    # Đọc file gốc và lọc theo threshold
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    filtered_lines = []
    for line in lines:
        parts = line.split()
        class_id = int(parts[1])  # Lấy class ID từ dòng
        conf = float(parts[-1])   # Lấy giá trị confidence score từ cuối dòng
        
        # Kiểm tra nếu confidence >= threshold tương ứng với class ID
        if conf >= thresholds.get(class_id, 0):  # Mặc định threshold = 0 nếu class ID không tồn tại
            filtered_lines.append(line)

    # Ghi kết quả đã lọc vào file mới
    with open(output_txt_path, 'w') as f:
        f.writelines(filtered_lines)

    print(f"Kết quả đã được lọc và lưu vào: {output_txt_path}")
