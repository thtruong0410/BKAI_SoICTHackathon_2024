import os
from predict import predict
from ultralytics import YOLO
from sahi_infer import sahi
from wbf import process_wbf
from post_process import post_process


prediction_files = []

data_path = "D:/blackbox_sourcecode_data/report2111/data/public_test"
output_path = "D:/blackbox_sourcecode_data/report2111/results"
os.makedirs(output_path, exist_ok=True)

#Step 3: WBF
wbf_output_file = os.path.join(output_path, "predict_fused.txt")
process_wbf(output_path, wbf_output_file)

# #Step 4: Post processing
final_output_file = os.path.join(output_path, "predict.txt")
output_path_area = os.path.join(output_path, "predict_delete_area_1280.txt")
output_path_ratio = os.path.join(output_path, "predict_delete_area_whratio_1280.txt")
post_process(wbf_output_file, data_path, output_path_area, output_path_ratio, final_output_file)