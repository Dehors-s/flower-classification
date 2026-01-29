#!/usr/bin/env python3
"""
花卉分类模型预测脚本 - 兼容官方格式版

使用方法:
    python predict.py <测试集文件夹> <输出文件路径>

示例:
    python predict.py ./unified_flower_dataset/images/test ./results/submission.csv
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# 所有可能的类别ID (原始类别标签)
CATEGORY_IDS = [
    # 164-245 范围
    164, 165, 166, 167, 169, 171, 172, 173, 174, 176, 177, 178, 179, 180,
    183, 184, 185, 186, 188, 189, 190, 192, 193, 194, 195, 197, 198, 199,
    200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
    214, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228,
    229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
    243, 244, 245,
    # 1734-1833 范围
    1734, 1743, 1747, 1749, 1750, 1751, 1759, 1765, 1770, 1772, 1774, 1776,
    1777, 1780, 1784, 1785, 1786, 1789, 1796, 1797, 1801, 1805, 1806, 1808,
    1818, 1827, 1833
]

# 模型类别数量
NUM_CLASSES = len(CATEGORY_IDS)


def detect_dataset_structure(test_dir):
    """检测数据集结构"""
    print(f"检测数据集结构: {test_dir}")

    # 检查目录是否存在
    if not os.path.exists(test_dir):
        print(f"❌ 测试集目录不存在: {test_dir}")
        return None

    items = os.listdir(test_dir)

    # 检查是否有数字命名的子文件夹（按类别分文件夹的结构）
    category_folders = []
    for item in items:
        item_path = os.path.join(test_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            category_folders.append(item)

    if category_folders:
        print(f"✅ 检测到按类别分文件夹的结构，找到 {len(category_folders)} 个类别文件夹")
        return "category_folders"
    else:
        # 检查是否有图片文件（所有图片在一个文件夹的结构）
        image_files = []
        for item in items:
            item_path = os.path.join(test_dir, item)
            if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(item)

        if image_files:
            print(f"✅ 检测到所有图片在一个文件夹的结构，找到 {len(image_files)} 张图片")
            return "single_folder"
        else:
            print("❌ 未找到任何图片文件")
            return None


def get_image_files_compatible(test_dir, img_extensions=None):
    """兼容官方格式的图片文件获取函数"""
    if img_extensions is None:
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

    image_files = []

    # 检测数据集结构
    structure = detect_dataset_structure(test_dir)

    if structure == "category_folders":
        # 按类别分文件夹的结构：递归搜索所有子文件夹
        print("正在递归搜索所有类别文件夹...")
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in img_extensions):
                    # 获取相对路径，保持文件夹结构信息
                    rel_path = os.path.relpath(os.path.join(root, file), test_dir)
                    image_files.append(rel_path)

        print(f"在类别文件夹结构中找到 {len(image_files)} 张图片")

    elif structure == "single_folder":
        # 所有图片在一个文件夹的结构
        print("正在扫描当前文件夹...")
        for file in os.listdir(test_dir):
            if any(file.lower().endswith(ext) for ext in img_extensions):
                image_files.append(file)

        print(f"在单文件夹结构中找到 {len(image_files)} 张图片")

    else:
        print("❌ 无法确定数据集结构")
        return []

    # 按文件名排序确保一致性
    image_files.sort()
    return image_files


def get_image_path(test_dir, filename):
    """根据文件名获取完整的图片路径，处理不同的目录结构"""
    # 首先尝试直接路径
    direct_path = os.path.join(test_dir, filename)
    if os.path.exists(direct_path):
        return direct_path

    # 如果直接路径不存在，可能是相对路径（在子文件夹中）
    # 递归搜索文件
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file == filename:
                return os.path.join(root, file)

    # 如果还是找不到，尝试使用基本文件名（不带路径）
    base_name = os.path.basename(filename)
    direct_path = os.path.join(test_dir, base_name)
    if os.path.exists(direct_path):
        return direct_path

    print(f"❌ 无法找到图片文件: {filename}")
    return None


# ==================== 模型定义部分保持不变 ====================
class FixedTorchViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 dim=384, depth=6, heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Patch嵌入
        self.patch_embed = nn.Conv2d(
            3, dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 类别token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))

        self.dropout = nn.Dropout(dropout)

        # Transformer层
        mlp_dim = int(dim * mlp_ratio)
        self.encoder_layers = nn.ModuleList([
            FixedTransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        # 层归一化
        self.norm = nn.LayerNorm(dim)

        # 分类头
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape

        # 使用卷积进行patch嵌入
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        # 添加类别token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 添加位置编码
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Transformer编码器
        for layer in self.encoder_layers:
            x = layer(x)

        # 分类
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)

        return x


class FixedTransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 自注意力
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = residual + self.dropout(attn_output)

        # MLP
        residual = x
        x = self.norm2(x)
        mlp_output = self.mlp(x)
        x = residual + self.dropout(mlp_output)

        return x


def debug_model_weights(model):
    """调试模型权重"""
    print("\n=== 模型权重调试信息 ===")

    if hasattr(model, 'head'):
        head_weight = model.head.weight.data
        head_bias = model.head.bias.data

        print(f"分类头权重形状: {head_weight.shape}")
        print(f"分类头偏置形状: {head_bias.shape}")
        print(f"分类头权重范数: {torch.norm(head_weight):.6f}")
        print(f"分类头偏置范数: {torch.norm(head_bias):.6f}")
        print(f"分类头权重范围: [{head_weight.min():.6f}, {head_weight.max():.6f}]")

    if hasattr(model, 'patch_embed'):
        patch_weight = model.patch_embed.weight.data
        print(f"Patch嵌入权重形状: {patch_weight.shape}")
        print(f"Patch嵌入权重范数: {torch.norm(patch_weight):.6f}")


def load_model_correctly(model_path, num_classes=NUM_CLASSES, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """正确加载模型"""
    print(f"正在加载模型: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None

    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"模型文件大小: {file_size:.2f} MB")

    if file_size < 1:
        print("❌ 模型文件可能已损坏，文件大小异常")
        return None

    model_config = {
        'image_size': 224,
        'patch_size': 16,
        'num_classes': num_classes,
        'dim': 384,
        'depth': 6,
        'heads': 8,
        'mlp_ratio': 4,
        'dropout': 0.1
    }

    model = FixedTorchViT(**model_config)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Checkpoint类型: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print("Checkpoint键:", list(checkpoint.keys()))

            state_dict = None
            for key in ['model_state_dict', 'state_dict', 'model', 'weights']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"使用键 '{key}' 加载状态字典")
                    break

            if state_dict is None:
                state_dict = checkpoint
                print("直接使用checkpoint作为状态字典")
        else:
            state_dict = checkpoint
            print("Checkpoint不是字典，直接作为状态字典使用")

        if state_dict:
            print(f"状态字典键数量: {len(state_dict)}")
            print("状态字典前10个键:")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                print(f"  {i + 1}. {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'}")

        if state_dict:
            try:
                model.load_state_dict(state_dict)
                print("✅ 严格匹配加载成功")
            except Exception as e:
                print(f"⚠️ 严格匹配失败: {e}")
                print("尝试非严格匹配...")

                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

                if missing_keys:
                    print(f"⚠️ 缺失的键: {len(missing_keys)}个")
                    for key in missing_keys[:5]:
                        print(f"    - {key}")
                    if len(missing_keys) > 5:
                        print(f"    ... 还有 {len(missing_keys) - 5} 个")

                if unexpected_keys:
                    print(f"⚠️ 意外的键: {len(unexpected_keys)}个")
                    for key in unexpected_keys[:5]:
                        print(f"    - {key}")
                    if len(unexpected_keys) > 5:
                        print(f"    ... 还有 {len(unexpected_keys) - 5} 个")

            debug_model_weights(model)

            model.to(device)
            model.eval()

            # 测试模型前向传播
            with torch.no_grad():
                test_input = torch.randn(1, 3, 224, 224).to(device)
                test_output = model(test_input)
                test_probs = torch.softmax(test_output, dim=1)
                max_prob = test_probs.max().item()

                print(f"测试前向传播 - 最大概率: {max_prob:.4f}")
                if max_prob < 0.1:
                    print("⚠️ 警告: 模型输出概率异常低，可能权重未正确加载")
                else:
                    print("✅ 模型前向传播测试正常")

            return model
        else:
            print("❌ 无法找到有效的状态字典")
            return None

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_transform():
    """创建图像预处理变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_images(model, image_files, test_dir, transform, device):
    """预测图片并返回结果"""
    predictions = []

    for i, filename in enumerate(image_files):
        try:
            # 使用兼容的路径获取方法
            img_path = get_image_path(test_dir, filename)
            if img_path is None:
                print(f"❌ 跳过无法找到的图片: {filename}")
                continue

            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()

            # 将预测索引映射到原始类别ID
            if predicted_idx < len(CATEGORY_IDS):
                category_id = CATEGORY_IDS[predicted_idx]
            else:
                category_id = CATEGORY_IDS[0]
                confidence = 0.1

            predictions.append({
                'filename': filename,
                'category_id': category_id,
                'confidence': confidence
            })

            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(image_files)} 张图片")

        except Exception as e:
            print(f"❌ 预测图片 {filename} 失败: {e}")
            predictions.append({
                'filename': filename,
                'category_id': CATEGORY_IDS[0],
                'confidence': 0.1
            })

    return predictions


def main():
    parser = argparse.ArgumentParser(description='花卉分类模型预测')

    parser.add_argument('test_img_dir', type=str, help='测试图片目录')
    parser.add_argument('output_path', type=str, help='预测结果输出路径 (CSV文件)')
    parser.add_argument('--model_path', type=str, default='pytorch_vit_model3.pth', help='模型文件路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')

    args = parser.parse_args()

    print(f'测试集目录: {args.test_img_dir}')
    print(f'输出文件: {args.output_path}')
    print(f'模型路径: {args.model_path}')
    print(f'运行设备: {args.device}')
    print(f'类别数量: {NUM_CLASSES}')
    print()

    # 检查路径
    if not os.path.exists(args.test_img_dir):
        print(f"错误: 测试集目录不存在: {args.test_img_dir}")
        return

    # 获取图片文件 - 使用兼容版本
    print("正在扫描测试集目录...")
    image_files = get_image_files_compatible(args.test_img_dir)

    if not image_files:
        print(f"错误: 在目录 {args.test_img_dir} 中未找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片")
    print()

    # 加载模型
    model = load_model_correctly(args.model_path, NUM_CLASSES, args.device)
    if model is None:
        print("❌ 模型加载失败，无法继续预测")
        return

    # 创建数据预处理
    transform = create_transform()

    # 进行预测
    print("正在进行预测...")
    predictions = predict_images(model, image_files, args.test_img_dir, transform, args.device)

    # 创建 DataFrame
    results_df = pd.DataFrame(predictions)
    results_df = results_df.sort_values('filename').reset_index(drop=True)

    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")

    # 保存结果
    results_df.to_csv(args.output_path, index=False)
    print(f"预测结果已保存到: {args.output_path}")
    print()

    # 显示详细统计
    if len(predictions) > 0:
        avg_confidence = results_df['confidence'].mean()
        max_confidence = results_df['confidence'].max()
        min_confidence = results_df['confidence'].min()

        print(f"预测完成!")
        print(f"总图片数: {len(predictions)}")
        print(f"平均置信度: {avg_confidence:.4f}")
        print(f"最高置信度: {max_confidence:.4f}")
        print(f"最低置信度: {min_confidence:.4f}")

        # 置信度分布
        conf_bins = [0, 0.1, 0.5, 0.7, 0.9, 1.0]
        conf_dist = pd.cut(results_df['confidence'], bins=conf_bins).value_counts().sort_index()
        print(f"\n置信度分布:")
        for bin_range, count in conf_dist.items():
            print(f"  {bin_range}: {count} 张图片")

        # 类别分布
        category_counts = results_df['category_id'].value_counts()
        print(f"\n预测类别分布 (前10个):")
        for category_id, count in category_counts.head(10).items():
            percentage = count / len(predictions) * 100
            print(f"  类别 {category_id}: {count} 张图片 ({percentage:.1f}%)")
    else:
        print("❌ 没有成功预测任何图片")


if __name__ == '__main__':
    main()