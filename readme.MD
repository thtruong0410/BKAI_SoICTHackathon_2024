# Hướng dẫn triển khai và xử lý dữ liệu

## 1. Khởi chạy môi trường
- **Bước 1:** Tại thư mục chứa `Dockerfile`, chạy lệnh sau để build image:
  ```bash
  docker build -t blackbox .
  ```
- **Bước 2:** Chạy container và truy cập vào container:
  ```bash
  docker run --gpus all --shm-size=100gb -it -v $(pwd):/workspace --name blackbox_cont blackbox
  ```
- **Bước 3:** Trong container, môi trường đã được cấu hình sẵn với đầy đủ các thư viện cần thiết cho quá trình training và inference.

---

## 2. Quá trình tiền xử lý dữ liệu
### 2.1 Chia dataset thành tập train và test
- Sử dụng script `source_code/split_train_test.py`:
  - **Dòng 7:** Đặt đường dẫn folder dữ liệu `daytime` và `nighttime` vào biến `dataset_dir`.
  - **Dòng 12:** Đặt đường dẫn thư mục YOLO dataset sau khi chia vào biến `yolo_dir`.

### 2.2 Huấn luyện mô hình YOLOv8l
- Tiến hành huấn luyện mô hình YOLOv8l để tạo model `yolov8l`.

### 2.3 Tiền xử lý dữ liệu với mô hình YOLOv8l
1. **Crop dữ liệu:**
    - Sử dụng file `source_code/crop_data.py`:
      - **Dòng 6:** Đặt đường dẫn folder dữ liệu gốc cần crop vào biến `data_path`.
      - **Dòng 52:** Đặt đường dẫn folder chứa dữ liệu sau khi crop vào biến `new_path`.
2. **Lựa chọn 6000 mẫu từ dữ liệu đã crop:**
    - Sử dụng file `source_code/get_dataset_from_cropdata.py`:
      - **Dòng 62:** Đặt đường dẫn folder chứa ảnh crop vào biến `image_dir`.
      - **Dòng 63:** Đặt đường dẫn folder chứa nhãn crop vào biến `label_dir`.
      - **Dòng 64:** Đặt đường dẫn folder chứa 6000 mẫu đã chọn vào biến `output_dir`.
3. **Gộp dữ liệu:**
    - Copy thủ công 6000 mẫu đã chọn vào dataset gốc để tạo bộ dữ liệu gộp.
4. **Bổ sung nhãn:**
    - Sử dụng file `source_code/create_label.py` để bổ sung nhãn:
      - **Dòng 72:** Đặt đường dẫn folder ảnh cần bổ sung nhãn vào biến `images_folder`.
      - **Dòng 73:** Đặt đường dẫn folder nhãn gốc vào biến `labels_folder`.
      - **Dòng 74:** Đặt đường dẫn folder lưu nhãn mới vào biến `labels_new_folder`.
      - **Dòng 75:** Đặt đường dẫn model YOLO dùng để bổ sung nhãn vào biến `model_path`.
5. **Augment dữ liệu:**
    - Sử dụng notebook `source_code/argument.ipynb` để tạo dữ liệu augment:
      - Thực hiện các thao tác xoay ảnh, tăng giảm độ sáng, và mô phỏng thời tiết mưa.

### 2.4 Tạo dataset hoàn chỉnh
- Gộp tất cả dữ liệu lại thành một folder dataset mới.

---

## 3. Quá trình huấn luyện
- Sử dụng file `report_final/source_code/train_yolo.py` để huấn luyện các mô hình yolo:
  - `yolov8x`, `yolo11x`.
- Sử dụng file `report_final/Co_DETR/train.sh` để huấn luyện mô hình Codetr
---

## 4. Quá trình xử lý kết quả
- Sau khi có model, tiến hành predict:
  1. **Predict yolo:** Sử dụng file `predict_yolo.py`.
  2. **Predict codetr:** Sử dụng file `predict_codetr.py`.
  3. **Wbf:** Sử dụng file `wbf.py`
  4. **Hậu xử lý kết quả:** 
      - Sử dụng file `post_process.py` để loại bỏ các box có tỷ lệ width-height không hợp lý.
      - Sử dụng file `filter_by_confidence.py` để lọc confidence theo từng class.
      - Sử dụng file `filter_by_poligon.py` để lọc mấy box nằm ở những vị trí không thể có object được

---

## 5. Tái tạo lại kết quả predict
1. Chạy container.
2. Chạy lệnh sau để tạo file predict:
   ```bash
   python3 source_code/main.py
   ```
3. Kết quả cuối cùng sẽ được lưu ở file result/no_area/predict_filtered.txt
