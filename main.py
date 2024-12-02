from ultralytics import YOLO

# Load a model
model = YOLO("/mnt/c/Users/admin/Desktop/code/yolo/ultralytics-main/ultralytics/cfg/models/v8/forest1.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/mnt/c/Users/admin/Desktop/code/yolo/ultralytics-main/ultralytics/cfg/datasets/forest.yaml", epochs=10, imgsz=640)
