from ultralytics import YOLO
model = YOLO("yolov8l.pt")
data_path = "report2111/data/data_default/dataset.yaml"
project_path = "./weights" 
name_train = "yolov8l"       
results = model.train(
    data=data_path,
    epochs=1024,
    project=project_path,
    name=name_train,
    batch=128,                    
    save_period=10,
    patience=50,
    cache=False,
    dropout=0.1,
    plots=True,
    degrees = 15
    )