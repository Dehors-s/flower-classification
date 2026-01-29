import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import random


def create_flower_dataset():
    """创建花卉数据集，符合训练脚本格式要求"""

    # 配置路径
    image_dir = r"D:\ptcharm\project\花卉分析\train_flower\train"
    label_file = r"D:\ptcharm\project\花卉分析\train_labels.csv"
    output_dir = r"D:\ptcharm\project\花卉分析"

    # 创建输出目录结构
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    # 清空并重新创建目录
    for dir_path in [train_dir, val_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    # 读取标签文件
    print("读取标签文件...")
    df = pd.read_csv(label_file)
    print(f"总图片数量: {len(df)}")

    # 统计类别信息
    category_counts = df['category_id'].value_counts().sort_index()
    print(f"类别数量: {len(category_counts)}")
    print("类别分布:")
    for category_id, count in category_counts.items():
        category_name = df[df['category_id'] == category_id]['chinese_name'].iloc[0]
        print(f"  类别 {category_id} ({category_name}): {count} 张图片")

    # 为每个类别创建训练集和验证集
    train_files = []
    val_files = []

    for category_id in df['category_id'].unique():
        # 获取该类别的所有图片
        category_files = df[df['category_id'] == category_id]['filename'].tolist()

        # 分割训练集和验证集 (80% 训练, 20% 验证)
        if len(category_files) > 1:
            train_cat, val_cat = train_test_split(
                category_files,
                test_size=0.2,
                random_state=42
            )
        else:
            # 如果只有一张图片，全部用于训练
            train_cat = category_files
            val_cat = []

        train_files.extend([(f, category_id) for f in train_cat])
        val_files.extend([(f, category_id) for f in val_cat])

    print(f"\n数据集分割完成:")
    print(f"训练集: {len(train_files)} 张图片")
    print(f"验证集: {len(val_files)} 张图片")

    # 复制文件到相应目录
    print("\n开始复制图片文件...")

    def copy_files(file_list, target_dir):
        """复制文件到目标目录"""
        copied_count = 0
        missing_count = 0

        for filename, category_id in file_list:
            src_path = os.path.join(image_dir, filename)
            category_dir = os.path.join(target_dir, str(category_id))

            # 创建类别目录
            os.makedirs(category_dir, exist_ok=True)

            # 复制文件
            if os.path.exists(src_path):
                dst_path = os.path.join(category_dir, filename)
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            else:
                print(f"警告: 文件不存在 {src_path}")
                missing_count += 1

        return copied_count, missing_count

    # 复制训练集
    print("复制训练集图片...")
    train_copied, train_missing = copy_files(train_files, train_dir)

    # 复制验证集
    print("复制验证集图片...")
    val_copied, val_missing = copy_files(val_files, val_dir)

    # 统计最终结果
    print("\n" + "=" * 50)
    print("数据集创建完成!")
    print("=" * 50)
    print(f"训练集: {train_copied} 张图片 (缺失: {train_missing})")
    print(f"验证集: {val_copied} 张图片 (缺失: {val_missing})")
    print(f"总复制: {train_copied + val_copied} 张图片")
    print(f"总缺失: {train_missing + val_missing} 张图片")

    # 验证数据集结构
    print("\n验证数据集结构...")
    train_categories = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    val_categories = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]

    print(f"训练集类别数: {len(train_categories)}")
    print(f"验证集类别数: {len(val_categories)}")

    # 统计每个类别的图片数量
    print("\n训练集类别分布:")
    for category in sorted(train_categories, key=int):
        category_path = os.path.join(train_dir, category)
        count = len([f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  类别 {category}: {count} 张图片")

    print("\n验证集类别分布:")
    for category in sorted(val_categories, key=int):
        category_path = os.path.join(val_dir, category)
        count = len([f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  类别 {category}: {count} 张图片")

    # 创建类别映射文件（可选）
    create_category_mapping(df, output_dir)

    return True


def create_category_mapping(df, output_dir):
    """创建类别映射文件"""
    mapping_file = os.path.join(output_dir, "category_mapping.csv")

    # 获取唯一的类别信息
    unique_categories = df[['category_id', 'chinese_name', 'english_name']].drop_duplicates()
    unique_categories = unique_categories.sort_values('category_id')

    # 添加训练用的映射ID（从0开始）
    unique_categories['mapped_id'] = range(len(unique_categories))

    # 保存映射文件
    unique_categories.to_csv(mapping_file, index=False, encoding='utf-8-sig')
    print(f"\n类别映射文件已保存: {mapping_file}")
    print("类别映射:")
    for _, row in unique_categories.iterrows():
        print(f"  原始ID: {row['category_id']} -> 映射ID: {row['mapped_id']} | {row['chinese_name']}")


def validate_dataset_structure():
    """验证数据集结构是否符合训练要求"""
    base_dir = r"D:\ptcharm\project\花卉分析"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    print("验证数据集结构...")

    def check_directory(dir_path, dir_name):
        if not os.path.exists(dir_path):
            print(f"❌ {dir_name}目录不存在: {dir_path}")
            return False

        categories = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        if not categories:
            print(f"❌ {dir_name}中没有找到类别目录")
            return False

        total_images = 0
        print(f"✅ {dir_name}结构:")
        for category in sorted(categories, key=lambda x: int(x) if x.isdigit() else x):
            category_path = os.path.join(dir_path, category)
            images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(images)
            print(f"  类别 {category}: {len(images)} 张图片")

        print(f"  {dir_name}总计: {len(categories)} 个类别, {total_images} 张图片")
        return True

    train_ok = check_directory(train_dir, "训练集")
    val_ok = check_directory(val_dir, "验证集")

    return train_ok and val_ok


if __name__ == "__main__":
    print("花卉数据集创建脚本")
    print("=" * 60)

    # 检查是否已经存在数据集
    if validate_dataset_structure():
        response = input("\n数据集已存在，是否重新创建? (y/n): ")
        if response.lower() != 'y':
            print("退出脚本")
            exit()

    # 创建数据集
    try:
        success = create_flower_dataset()
        if success:
            print("\n" + "=" * 60)
            print("✅ 数据集创建成功!")
            print("=" * 60)

            # 最终验证
            validate_dataset_structure()

            print("\n现在可以运行训练脚本了!")
            print("训练命令: python train_v1.py")
            print("测试命令: python VIT_test.py")
        else:
            print("❌ 数据集创建失败!")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback

        traceback.print_exc()