import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.io import DataLoader, Dataset
from paddle.optimizer import AdamW
import os
import numpy as np
import time
import pandas as pd
from PIL import Image

print(f"PaddlePaddle版本: {paddle.__version__}")


# 优化配置参数
class Config:
    data_dir = '/home/aistudio/work/flower_dataset'
    num_classes = 100
    image_size = 224
    batch_size = 32
    learning_rate = 1e-4  # 使用较小的学习率
    weight_decay = 1e-4
    num_epochs = 50
    save_dir = '/home/aistudio/work/output_vit_fixed'
    log_interval = 20

    os.makedirs(save_dir, exist_ok=True)


def count_parameters(model):
    """统计模型参数量"""
    total_params = 0
    for param in model.parameters():
        total_params += int(param.numel())
    return total_params


# 修复的ViT模型
class FixedViT(nn.Layer):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 dim=384, depth=6, heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        # Patch嵌入 - 使用Conv2D而不是Linear，更稳定
        self.patch_embed = nn.Conv2D(
            3, dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 类别token和位置编码
        self.cls_token = self.create_parameter(
            shape=[1, 1, dim],
            default_initializer=nn.initializer.TruncatedNormal(std=0.02)
        )
        self.pos_embedding = self.create_parameter(
            shape=[1, self.num_patches + 1, dim],
            default_initializer=nn.initializer.TruncatedNormal(std=0.02)
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer层
        mlp_dim = int(dim * mlp_ratio)
        self.encoder_layers = nn.LayerList([
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        # 层归一化
        self.norm = nn.LayerNorm(dim)

        # 分类头
        self.head = nn.Linear(dim, num_classes)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.initializer.TruncatedNormal(std=0.02)(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.initializer.Constant(0)(m.bias)
            nn.initializer.Constant(1.0)(m.weight)
        elif isinstance(m, nn.Conv2D):
            nn.initializer.TruncatedNormal(std=0.02)(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(0)(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape

        # 使用卷积进行patch嵌入
        x = self.patch_embed(x)  # [B, dim, H//P, W//P]
        x = x.flatten(2)  # [B, dim, num_patches]
        x = x.transpose([0, 2, 1])  # [B, num_patches, dim]

        # 添加类别token
        cls_tokens = self.cls_token.expand([B, -1, -1])
        x = paddle.concat([cls_tokens, x], axis=1)

        # 添加位置编码
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Transformer编码器
        for layer in self.encoder_layers:
            x = layer(x)

        # 分类
        x = self.norm(x)
        x = x[:, 0]  # 取类别token
        x = self.head(x)

        return x


class TransformerBlock(nn.Layer):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(dim, heads, dropout=dropout)
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
        attn_output = self.attn(x, x, x)
        x = residual + self.dropout(attn_output)

        # MLP
        residual = x
        x = self.norm2(x)
        mlp_output = self.mlp(x)
        x = residual + self.dropout(mlp_output)

        return x


class FlowerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []

        # 读取类别
        categories = sorted([d for d in os.listdir(data_dir) if d.isdigit()], key=int)
        self.label_mapping = {int(cat): idx for idx, cat in enumerate(categories)}

        print(f"加载数据集: {data_dir}")
        print(f"发现类别: {len(categories)}个")

        for category in categories:
            category_dir = os.path.join(data_dir, category)
            if os.path.isdir(category_dir):
                for file in os.listdir(category_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append(os.path.join(category_dir, file))
                        self.labels.append(self.label_mapping[int(category)])

        print(f"图片数量: {len(self.samples)}")

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"加载图片失败: {img_path}, 错误: {e}")
            dummy_img = paddle.zeros([3, Config.image_size, Config.image_size])
            return dummy_img, label

    def __len__(self):
        return len(self.samples)


def create_transforms():
    image_size = Config.image_size

    train_transforms = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(0.3),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


def create_model():
    # 使用中等大小的ViT模型
    model = FixedViT(
        image_size=Config.image_size,
        patch_size=16,
        num_classes=Config.num_classes,
        dim=384,  # 中等维度
        depth=6,  # 中等深度
        heads=8,  # 中等头数
        mlp_ratio=4,  # MLP扩展比例
        dropout=0.1
    )

    print(f"创建修复的ViT模型")
    print(f"输入尺寸: {Config.image_size}")
    print(f"类别数: {Config.num_classes}")

    total_params = count_parameters(model)
    print(f"总参数量: {total_params:,}")

    return model


def test_model_forward():
    """测试模型前向传播"""
    print("测试模型前向传播...")
    model = create_model()

    # 创建测试数据
    test_input = paddle.randn([2, 3, Config.image_size, Config.image_size])

    # 前向传播
    with paddle.no_grad():
        output = model(test_input)

    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # 测试损失计算
    criterion = nn.CrossEntropyLoss()
    test_target = paddle.randint(0, Config.num_classes, [2])
    loss = criterion(output, test_target)
    print(f"测试损失: {loss.item():.4f}")

    # 检查梯度
    model.train()
    test_output = model(test_input)
    test_loss = criterion(test_output, test_target)
    test_loss.backward()

    # 检查梯度是否存在
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = paddle.norm(param.grad).item()
            grad_norms.append(grad_norm)

    if grad_norms:
        print(f"梯度范数范围: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
    else:
        print("❌ 没有检测到梯度")

    return model


def train_fixed_vit():
    """训练修复的ViT模型"""
    print("=" * 60)
    print("开始修复的ViT训练")
    print("=" * 60)

    # 设置随机种子
    paddle.seed(42)
    np.random.seed(42)

    # 数据目录
    train_dir = os.path.join(Config.data_dir, 'train')
    val_dir = os.path.join(Config.data_dir, 'val')

    # 创建数据变换
    train_transforms, val_transforms = create_transforms()

    # 创建数据集
    print("加载训练集...")
    train_dataset = FlowerDataset(train_dir, transform=train_transforms)
    print("加载验证集...")
    val_dataset = FlowerDataset(val_dir, transform=val_transforms)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=2
    )

    print(f"\n数据集统计:")
    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")
    print(f"批次大小: {Config.batch_size}")

    # 测试模型
    model = test_model_forward()

    # 优化器和损失函数
    optimizer = AdamW(
        parameters=model.parameters(),
        learning_rate=Config.learning_rate,
        weight_decay=Config.weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    # 训练记录
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_accuracy = 0.0

    print("\n开始训练...")
    print("=" * 60)

    for epoch in range(1, Config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{Config.num_epochs}")
        print("-" * 50)

        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.clear_grad()

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pred = output.argmax(axis=1)
            correct += (pred == target).sum().item()
            total += target.shape[0]

            if batch_idx % Config.log_interval == 0:
                current_lr = optimizer.get_lr()
                batch_acc = 100. * (pred == target).astype('float32').mean().item()
                print(
                    f'训练 [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {batch_acc:.1f}% LR: {current_lr:.2e}')

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_time = time.time() - start_time

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        print(f'训练结果 - 损失: {avg_train_loss:.4f}, 准确率: {train_accuracy:.2f}%, 时间: {train_time:.1f}秒')

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with paddle.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                pred = output.argmax(axis=1)
                val_correct += (pred == target).sum().item()
                val_total += target.shape[0]

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f'验证结果 - 损失: {avg_val_loss:.4f}, 准确率: {val_accuracy:.2f}%')

        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_path = os.path.join(Config.save_dir, 'best_model.pdparams')
            paddle.save(model.state_dict(), best_model_path)
            print(f"🚀 保存最佳模型，准确率: {val_accuracy:.2f}%")

        # 每5个epoch保存一次检查点
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(Config.save_dir, f'checkpoint_epoch_{epoch}.pdparams')
            paddle.save(model.state_dict(), checkpoint_path)
            print(f"💾 保存检查点: {checkpoint_path}")

        # 提前停止检查 - 放宽条件
        if epoch >= 5 and best_accuracy < 3.0:
            print("❌ 准确率过低，停止训练")
            break
        elif epoch >= 5 and avg_train_loss > 4.0 and train_accuracy < 5.0:
            print("❌ 损失没有下降，停止训练")
            break

    # 保存最终模型
    final_model_path = os.path.join(Config.save_dir, 'final_model.pdparams')
    paddle.save(model.state_dict(), final_model_path)
    print(f"💾 保存最终模型: {final_model_path}")

    # 训练总结
    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print(f"🏆 最佳验证准确率: {best_accuracy:.2f}%")
    print("=" * 60)

    # 保存训练历史
    history = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    })
    history_path = os.path.join(Config.save_dir, 'training_history.csv')
    history.to_csv(history_path, index=False)
    print(f"📈 训练历史已保存: {history_path}")

    return model, history


if __name__ == "__main__":
    model, history = train_fixed_vit()