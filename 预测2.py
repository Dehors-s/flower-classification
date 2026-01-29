#!/usr/bin/env python3
"""
修复版预测脚本 - 兼容官方环境
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import json

print("=== 预测脚本开始执行 ===")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 尝试导入自定义模块
try:
    from model import FixedViT

    print("✅ 成功导入模型模块")
except ImportError as e:
    print(f"❌ 导入模型模块失败: {e}")
    sys.exit(1)


def load_model_safe():
    """安全加载模型"""
    try:
        # 模型配置文件路径
        config_path = './model/config.json'
        model_path = './model/best_model.pth'

        print(f"配置文件路径: {config_path}")
        print(f"模型文件路径: {model_path}")

        # 检查文件是否存在
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return None

        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return None

        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ 加载配置: {config}")

        # 创建模型
        model = FixedViT(
            image_size=config.get('image_size', 224),
            patch_size=config.get('patch_size', 16),
            num_classes=config.get('num_classes', 100),
            dim=config.get('dim', 384),
            depth=config.get('depth', 6),
            heads=config.get('heads', 8),
            mlp_ratio=config.get('mlp_ratio', 4),
            dropout=config.get('dropout', 0.1)
        )
        print("✅ 模型创建成功")

        # 加载权重
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        print("✅ 模型权重加载成功")

        return model, config, device

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def create_simple_transform(image_size=224):
    """创建简单的数据变换"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def find_image_files(test_dir):
    """查找所有图片文件"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    image_files = []

    print(f"搜索目录: {test_dir}")

    if not os.path.exists(test_dir):
        print(f"❌ 测试目录不存在: {test_dir}")
        return []

    # 递归搜索所有图片文件
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                rel_path = os.path.relpath(os.path.join(root, file), test_dir)
                image_files.append(rel_path)

    print(f"找到 {len(image_files)} 张图片")
    return sorted(image_files)


def predict_single_image(model, image_path, transform, device):
    """预测单张图片"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence

    except Exception as e:
        print(f"❌ 预测图片失败 {image_path}: {e}")
        return 0, 0.0  # 返回默认值


def main():
    parser = argparse.ArgumentParser(description='花卉分类预测')
    parser.add_argument('test_dir', type=str, help='测试图片目录')
    parser.add_argument('output_path', type=str, help='输出CSV文件路径')

    args = parser.parse_args()

    print(f"测试目录: {args.test_dir}")
    print(f"输出路径: {args.output_path}")

    # 检查测试目录
    if not os.path.exists(args.test_dir):
        print(f"❌ 测试目录不存在: {args.test_dir}")
        sys.exit(1)

    # 加载模型
    model, config, device = load_model_safe()
    if model is None:
        print("❌ 无法加载模型，退出")
        sys.exit(1)

    # 创建数据变换
    image_size = config.get('image_size', 224)
    transform = create_simple_transform(image_size)

    # 查找图片文件
    image_files = find_image_files(args.test_dir)
    if not image_files:
        print("❌ 未找到任何图片文件")
        sys.exit(1)

    # 进行预测
    results = []
    print("开始预测...")

    for i, filename in enumerate(image_files):
        if i % 10 == 0:
            print(f"进度: {i + 1}/{len(image_files)}")

        img_path = os.path.join(args.test_dir, filename)
        predicted_class, confidence = predict_single_image(model, img_path, transform, device)

        results.append({
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    # 保存结果
    df = pd.DataFrame(results)

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_csv(args.output_path, index=False)

    print(f"✅ 预测完成! 结果保存到: {args.output_path}")
    print(f"处理图片数量: {len(results)}")


if __name__ == '__main__':
    main()