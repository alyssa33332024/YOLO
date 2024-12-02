import cv2
import os

def draw_yolo_boxes(image_path, label_path, class_names, save_root):
    """
    可视化YOLO标注框
    :param image_path: 图片路径
    :param label_path: 对应的YOLO标注文件路径 (.txt)
    :param class_names: 类别名称列表（可选）
    """
    # 读取图像
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # 读取YOLO格式的标注文件
    with open(label_path, 'r') as f:
        labels = f.readlines()

    for label in labels:
        # 每行格式为: class_id x_center y_center width height
        parts = label.strip().split(' ')
        class_id = int(parts[0])
        x_center, y_center, w, h = map(float, parts[1:])

        # 将YOLO的归一化坐标转换为实际坐标
        x_center = int(x_center * width)
        y_center = int(y_center * height)
        w = int(w * width)
        h = int(h * height)

        # 计算矩形框的左上角和右下角坐标
        x1 = x_center - w // 2
        y1 = y_center - h // 2
        x2 = x_center + w // 2
        y2 = y_center + h // 2

        # 绘制矩形框
        colors = [(0, 255, 0),(0,0,255),(255,255,255)] # 绿色，表示标注框
        thickness = 2
        image = cv2.rectangle(image, (x1, y1), (x2, y2), colors[class_id], thickness)

        # 如果有类别名称，可以在框上添加标签
        if class_names:
            label_text = class_names[class_id] if class_id < len(class_names) else str(class_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            cv2.putText(image, label_text, (x1, y1 - 5), font, font_scale, colors[class_id], font_thickness)

    # 显示图像
    # cv2.imshow('Image with YOLO boxes', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存标注框后的图片（如果需要）
    fn= image_path.split("/")[-1]
    cv2.imwrite(os.path.join(save_root,fn), image)

# 示例
image_root = '/mnt/c/Users/alyss/Desktop/yolo/datasets/dataset_last/images/test'  # 图片路径
label_root = '/mnt/c/Users/alyss/Desktop/yolo/datasets/dataset_last/labels/test'  # 对应的标注文件路径
save_root ='/mnt/c/Users/alyss/Desktop/yolo/vis'

for fn in os.listdir(image_root):
    image_path=os.path.join(image_root,fn)
    label_path=os.path.join(label_root,fn.split(".")[0]+".txt")


# 如果有类别名称，可以传入一个类别名称列表
    class_names = ['fire', 'smoke']  # 示例类别名称
    draw_yolo_boxes(image_path, label_path, class_names,save_root)
