import os
import zipfile
import math
from pathlib import Path


def split_folder_to_zips(folder_path, output_dir, max_size_mb=400):
    """
    将文件夹中的图片拆分成多个小于指定大小的压缩包

    参数:
        folder_path: 源文件夹路径
        output_dir: 输出目录
        max_size_mb: 每个压缩包的最大大小(MB)
    """
    # 转换为Path对象
    source_folder = Path(folder_path)
    output_dir = Path(output_dir)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.raw', '.heic'}

    # 收集所有图片文件
    image_files = []
    total_size = 0

    print("正在扫描图片文件...")
    for file_path in source_folder.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            file_size = file_path.stat().st_size
            image_files.append((file_path, file_size))
            total_size += file_size

    print(f"找到 {len(image_files)} 个图片文件")
    print(f"总大小: {total_size / 1024 / 1024 / 1024:.2f} GB")

    # 按原顺序排序（保持文件顺序）
    # 如果您希望按其他方式排序，可以修改这里
    # 例如：按文件名排序：image_files.sort(key=lambda x: x[0].name)
    # 或者按大小排序：image_files.sort(key=lambda x: x[1])

    # 拆分文件到不同的组
    max_size_bytes = max_size_mb * 1024 * 1024
    part_num = 1
    current_size = 0
    current_files = []

    print("\n开始拆分和压缩...")

    for file_path, file_size in image_files:
        # 如果单个文件就超过限制（不太可能，但安全起见）
        if file_size > max_size_bytes:
            print(f"警告: 文件 {file_path.name} 大小 {file_size / 1024 / 1024:.2f}MB 超过单个压缩包限制，将单独压缩")
            # 单独创建一个压缩包
            zip_path = output_dir / f"{source_folder.name}_part{part_num}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(file_path, file_path.relative_to(source_folder))
            print(f"创建 part{part_num}: 单个文件 {file_path.name}")
            part_num += 1
            continue

        # 如果添加当前文件会超过限制，创建新的压缩包
        if current_size + file_size > max_size_bytes and current_files:
            # 保存当前压缩包
            create_zip_part(source_folder, output_dir, source_folder.name, part_num, current_files)
            print(f"创建 part{part_num}: {len(current_files)} 个文件, {current_size / 1024 / 1024:.2f}MB")

            # 重置计数器
            part_num += 1
            current_size = 0
            current_files = []

        # 添加文件到当前组
        current_files.append((file_path, file_size))
        current_size += file_size

    # 保存最后一个压缩包
    if current_files:
        create_zip_part(source_folder, output_dir, source_folder.name, part_num, current_files)
        print(f"创建 part{part_num}: {len(current_files)} 个文件, {current_size / 1024 / 1024:.2f}MB")

    # 计算预计的压缩包数量
    estimated_parts = math.ceil(total_size / max_size_bytes)
    print(f"\n拆分完成！")
    print(f"原文件夹: {source_folder}")
    print(f"图片数量: {len(image_files)}")
    print(f"总大小: {total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"生成的压缩包数量: {part_num}")
    print(f"输出目录: {output_dir}")


def create_zip_part(source_folder, output_dir, base_name, part_num, file_list):
    """创建单个压缩包部分"""
    zip_path = output_dir / f"{base_name}_part{part_num:02d}.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path, file_size in file_list:
            # 保持相对路径结构
            arcname = file_path.relative_to(source_folder)
            zipf.write(file_path, arcname)


def verify_split(source_folder, output_dir):
    """验证拆分是否完整（可选功能）"""
    print("\n正在验证文件完整性...")

    source_folder = Path(source_folder)
    output_dir = Path(output_dir)

    # 收集源文件夹中的所有图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.raw', '.heic'}
    source_files = set()

    for file_path in source_folder.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            source_files.add(file_path.relative_to(source_folder))

    # 收集所有压缩包中的文件
    zip_files = set()
    for zip_path in output_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            for name in zipf.namelist():
                # 跳过目录
                if not name.endswith('/'):
                    zip_files.add(Path(name))

    # 比较差异
    missing_files = source_files - zip_files
    extra_files = zip_files - source_files

    if not missing_files and not extra_files:
        print("✓ 验证通过！所有文件都已正确压缩")
        print(f"  源文件数量: {len(source_files)}")
        print(f"  压缩包中文件数量: {len(zip_files)}")
    else:
        print("✗ 验证失败！")
        if missing_files:
            print(f"  缺失文件: {len(missing_files)} 个")
            for f in list(missing_files)[:10]:  # 只显示前10个
                print(f"    - {f}")
        if extra_files:
            print(f"  多余文件: {len(extra_files)} 个")

    return len(missing_files) == 0


# 使用示例
if __name__ == "__main__":
    # 配置参数
    SOURCE_FOLDER = r"D:\ptcharm\project\花卉分析\train"  # 替换为您的图片文件夹路径
    OUTPUT_DIR = r"D:\ptcharm\project\花卉分析\train_zips"  # 替换为输出目录
    MAX_SIZE_MB = 400  # 每个压缩包最大大小(MB)

    # 执行拆分
    split_folder_to_zips(SOURCE_FOLDER, OUTPUT_DIR, MAX_SIZE_MB)

    # 可选：验证完整性
    verify_result = input("\n是否要验证文件完整性？(y/n): ")
    if verify_result.lower() == 'y':
        verify_split(SOURCE_FOLDER, OUTPUT_DIR)