import os
import shutil

# 分割训练集测试集

path = "15-Scene"
train_path = "train"
test_path = "test"
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

for root, dirs, files in os.walk(path):
    if root == path:
        # 只处理第一层子目录
        continue
    class_name = os.path.basename(root)
    # 创建当前类别在训练集和测试集中的文件夹
    train_class_path = os.path.join(train_path, class_name)
    test_class_path = os.path.join(test_path, class_name)
    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(test_class_path, exist_ok=True)
    
    image_files = [f for f in files if f.endswith(".jpg")]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    train_files = image_files[:150]
    test_files = image_files[150:]
    
    # 移动文件到训练集和测试集文件夹
    for f in train_files:
        shutil.move(os.path.join(root, f), os.path.join(train_class_path, f))
    for f in test_files:
        shutil.move(os.path.join(root, f), os.path.join(test_class_path, f))