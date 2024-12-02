from ultralytics import YOLO
import cv2,os

# 加载模型
exp_index='test2'
model = YOLO("/mnt/c/Users/admin/Desktop/code/yolo/runs/detect/train44/weights/best.pt")


# 评估模型在验证集上的性能
model.val(data="/mnt/c/Users/admin/Desktop/code/yolo/ultralytics-main/ultralytics/cfg/datasets/forest.yaml")

# 在图像上执行对象检测
image_root='/mnt/c/Users/admin/Desktop/code/yolo/datasets/dataset_last/images/test'
output_root=f'/mnt/c/Users/admin/Desktop/code/yolo/pred_vis/{exp_index}'
os.makedirs(output_root,exist_ok=True)
image_list=[]
saved_image_nums=10

for fn in os.listdir(image_root)[:saved_image_nums]:
    image_path=os.path.join(image_root,fn)
    image_list.append(image_path)
results = model(image_list)
# return a list of Results objects

# Process results list
for result, fn in zip(results,image_list):
    save_name= os.path.join(output_root,fn.split('/')[-1])
    boxes = result.boxes # Boxes object for bounding box outputs
    masks = result.masks # Masks object for segmentation masks outputs
    keypoints = result.keypoints # Keypoints object for pose outputs
    probs = result.probs # Probs object for classification outputs
    obb = result.obb # Oriented boxes object for OBB outputs
     # result.show()  # display to screen
    result.save(filename=save_name) # save to disk




