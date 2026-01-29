import os
from PIL import Image
import numpy as np
from torchvision import transforms
import random
import shutil


def augment_images(input_dir, output_dir, augmentations_per_image=4):
    """
    对输入目录中的图像进行增强，并保存到输出目录

    参数:
        input_dir (str): 输入目录路径（包含按类别分组的图像）
        output_dir (str): 输出目录路径
        augmentations_per_image (int): 每张原始图像生成的增强版本数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 定义增强变换
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    ])

    # 遍历所有类别文件夹
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        if not os.path.isdir(class_path):
            continue

        # 创建输出类别目录
        output_class_path = os.path.join(output_dir, class_dir)
        os.makedirs(output_class_path, exist_ok=True)

        print(f"处理类别: {class_dir}")

        # 遍历类别中的所有图像
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            img_path = os.path.join(class_path, img_file)

            try:
                # 打开原始图像
                img = Image.open(img_path).convert('RGB')

                # 保存原始图像到输出目录
                base_name = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1]

                original_output_path = os.path.join(output_class_path, f"{base_name}_original{ext}")
                img.save(original_output_path)

                # 生成增强图像
                for i in range(augmentations_per_image):
                    augmented_img = augmentation_transforms(img)

                    # 生成增强图像的文件名
                    augmented_output_path = os.path.join(output_class_path, f"{base_name}_aug{i + 1}{ext}")

                    # 保存增强图像
                    augmented_img.save(augmented_output_path)

            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")

    print(f"图像增强完成! 增强后的图像保存在: {output_dir}")


def copy_val_test_data(source_dir, target_dir):
    """
    复制验证集和测试集数据（不进行增强）

    参数:
        source_dir (str): 源目录路径
        target_dir (str): 目标目录路径
    """
    if os.path.exists(source_dir):
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        print(f"已复制 {source_dir} 到 {target_dir}")
    else:
        print(f"警告: 目录 {source_dir} 不存在")


if __name__ == "__main__":
    # 设置路径
    train_dir = "./train"  # 原始训练集目录
    val_dir = "./val"  # 验证集目录
    test_dir = "./test"  # 测试集目录

    augmented_train_dir = "./train_augmented"  # 增强后的训练集目录
    augmented_val_dir = "./val_augmented"  # 增强后的验证集目录
    augmented_test_dir = "./test_augmented"  # 增强后的测试集目录

    # 对训练集进行增强
    print("开始对训练集图像进行增强...")
    augment_images(train_dir, augmented_train_dir, augmentations_per_image=4)

    # 复制验证集和测试集（不增强）
    print("\n复制验证集和测试集数据...")
    copy_val_test_data(val_dir, augmented_val_dir)
    copy_val_test_data(test_dir, augmented_test_dir)

    print("\n所有操作完成!")
    print(f"增强后的训练集: {augmented_train_dir}")
    print(f"验证集: {augmented_val_dir}")
    print(f"测试集: {augmented_test_dir}")