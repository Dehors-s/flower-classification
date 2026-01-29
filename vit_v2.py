import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.io import DataLoader, Dataset
from paddle.optimizer import AdamW
from paddle.optimizer.lr import CosineAnnealingDecay
import os
import numpy as np
import time
import pandas as pd
from PIL import Image
import math

print(f"PaddlePaddle版本: {paddle.__version__}")


# 优化配置参数
class Config:
    data_dir = '/home/aistudio/work/flower_dataset'
    num_classes = 100
    image_size = 224
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-4
    num_epochs = 100
    save_dir = '/home/aistudio/work/output_vit_improved'
    log_interval = 20
    resume_checkpoint = None

    # 调整正则化参数
    dropout_rate = 0.1
    stochastic_depth_rate = 0.05

    os.makedirs(save_dir, exist_ok=True)


def count_parameters(model):
    """统计模型参数量"""
    total_params = 0
    for param in model.parameters():
        total_params += int(param.numel())
    return total_params


# 标签平滑交叉熵损失
class LabelSmoothCrossEntropyLoss(nn.Layer):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = nn.functional.log_softmax(pred, axis=-1)
        nll_loss = -paddle.take_along_axis(log_prob, target.unsqueeze(1), axis=1)
        nll_loss = nll_loss.squeeze(1)

        smooth_loss = -log_prob.mean(axis=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# 添加随机深度（Stochastic Depth）
class StochasticDepth(nn.Layer):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training or self.drop_rate == 0:
            return x

        keep_prob = 1 - self.drop_rate
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        random_tensor = paddle.rand(shape, dtype=x.dtype) + keep_prob
        random_tensor = paddle.floor(random_tensor)

        return x / keep_prob * random_tensor


# 修复的ViT模型 - 简化架构
class ImprovedViT(nn.Layer):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 dim=384, depth=6, heads=8, mlp_ratio=4, dropout=0.1,
                 stochastic_depth_rate=0.1):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Patch嵌入 - 使用原始ViT的方法
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
            TransformerBlock(dim, heads, mlp_dim, dropout,
                             stochastic_depth_rate * (i / (depth - 1)) if depth > 1 else 0)
            for i in range(depth)
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
    def __init__(self, dim, heads, mlp_dim, dropout=0.1, stochastic_depth_rate=0.0):
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
        self.stochastic_depth = StochasticDepth(stochastic_depth_rate)

    def forward(self, x):
        # 自注意力 + 随机深度
        residual = x
        x = self.norm1(x)
        attn_output = self.attn(x, x, x)
        x = self.stochastic_depth(attn_output)
        x = residual + x

        # MLP + 随机深度
        residual = x
        x = self.norm2(x)
        mlp_output = self.mlp(x)
        x = self.stochastic_depth(mlp_output)
        x = residual + x

        return x


class FlowerDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
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
            dummy_img = paddle.rand([3, Config.image_size, Config.image_size])
            return dummy_img, label

    def __len__(self):
        return len(self.samples)


def create_transforms():
    """创建增强的数据变换"""
    image_size = Config.image_size

    train_transforms = T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


def create_model():
    """创建改进的ViT模型"""
    model = ImprovedViT(
        image_size=Config.image_size,
        patch_size=16,
        num_classes=Config.num_classes,
        dim=384,
        depth=6,
        heads=8,
        mlp_ratio=4,
        dropout=Config.dropout_rate,
        stochastic_depth_rate=Config.stochastic_depth_rate
    )

    print(f"创建改进的ViT模型")
    print(f"输入尺寸: {Config.image_size}")
    print(f"类别数: {Config.num_classes}")
    print(f"Dropout率: {Config.dropout_rate}")
    print(f"随机深度率: {Config.stochastic_depth_rate}")

    total_params = count_parameters(model)
    print(f"总参数量: {total_params:,}")

    return model


def create_optimizer_scheduler(model, train_loader):
    """创建优化器和学习率调度器"""
    learning_rate = Config.learning_rate

    # 创建优化器
    optimizer = AdamW(
        learning_rate=learning_rate,
        parameters=model.parameters(),
        weight_decay=Config.weight_decay
    )

    # 创建学习率调度器
    scheduler = CosineAnnealingDecay(
        learning_rate=learning_rate,
        T_max=Config.num_epochs * len(train_loader)
    )

    return optimizer, scheduler


def load_checkpoint(model, optimizer=None, scheduler=None):
    """加载检查点 - 修复权重不匹配问题"""
    if Config.resume_checkpoint and os.path.exists(Config.resume_checkpoint):
        print(f"加载检查点: {Config.resume_checkpoint}")
        checkpoint = paddle.load(Config.resume_checkpoint)

        # 检查检查点类型
        if 'model_state_dict' in checkpoint:
            checkpoint_state_dict = checkpoint['model_state_dict']
        else:
            checkpoint_state_dict = checkpoint

        # 获取当前模型的状态字典
        model_state_dict = model.state_dict()

        # 创建新的状态字典，只加载匹配的参数
        new_state_dict = {}
        loaded_params = 0
        skipped_params = 0

        for param_name in model_state_dict.keys():
            if param_name in checkpoint_state_dict:
                # 检查形状是否匹配
                if model_state_dict[param_name].shape == checkpoint_state_dict[param_name].shape:
                    new_state_dict[param_name] = checkpoint_state_dict[param_name]
                    loaded_params += 1
                else:
                    print(f"跳过参数 {param_name}: 形状不匹配")
                    new_state_dict[param_name] = model_state_dict[param_name]
                    skipped_params += 1
            else:
                print(f"初始化新参数: {param_name}")
                new_state_dict[param_name] = model_state_dict[param_name]
                skipped_params += 1

        # 加载状态字典
        model.set_state_dict(new_state_dict)
        print(f"参数加载完成: {loaded_params}个参数已加载, {skipped_params}个参数跳过/初始化")

        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.set_state_dict(checkpoint['optimizer_state_dict'])
                print("优化器状态已加载")
            except:
                print("优化器状态加载失败，使用默认优化器状态")

        # 加载调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.set_state_dict(checkpoint['scheduler_state_dict'])
                print("调度器状态已加载")
            except:
                print("调度器状态加载失败，使用默认调度器状态")

        # 加载训练状态
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        train_history = checkpoint.get('train_history', {})

        print(f"从epoch {start_epoch}继续训练, 最佳准确率: {best_accuracy:.2f}%")

        return start_epoch, best_accuracy, train_history
    else:
        if Config.resume_checkpoint:
            print(f"警告: 检查点 {Config.resume_checkpoint} 不存在，从头开始训练")
        return 1, 0.0, {}


def save_checkpoint(epoch, model, optimizer, scheduler, best_accuracy, train_history, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_accuracy': best_accuracy,
        'train_history': train_history,
        'config': {
            'image_size': Config.image_size,
            'num_classes': Config.num_classes,
            'batch_size': Config.batch_size,
        }
    }

    # 保存常规检查点
    checkpoint_path = os.path.join(Config.save_dir, f'checkpoint_epoch_{epoch}.pdparams')
    paddle.save(checkpoint, checkpoint_path)

    # 保存最佳模型
    if is_best:
        best_model_path = os.path.join(Config.save_dir, 'best_model.pdparams')
        paddle.save(checkpoint, best_model_path)
        print(f"🚀 保存最佳模型，准确率: {best_accuracy:.2f}%")

    # 保存最新模型
    latest_model_path = os.path.join(Config.save_dir, 'latest_model.pdparams')
    paddle.save(checkpoint, latest_model_path)

    return checkpoint_path


def train_improved_vit():
    """训练改进的ViT模型"""
    print("=" * 60)
    print("开始改进的ViT训练")
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
    train_dataset = FlowerDataset(train_dir, transform=train_transforms, is_train=True)
    print("加载验证集...")
    val_dataset = FlowerDataset(val_dir, transform=val_transforms, is_train=False)

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

    # 创建模型
    model = create_model()

    # 创建优化器和调度器
    optimizer, scheduler = create_optimizer_scheduler(model, train_loader)

    # 加载检查点
    start_epoch, best_accuracy, train_history = load_checkpoint(model, optimizer, scheduler)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练记录
    if not train_history:
        train_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }

    print("\n开始训练...")
    print("=" * 60)

    for epoch in range(start_epoch, Config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{Config.num_epochs}")
        print("-" * 50)

        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            current_step = (epoch - 1) * len(train_loader) + batch_idx

            optimizer.clear_grad()

            output = model(data)
            loss = criterion(output, target)

            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # 计算准确率
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

        train_history['train_loss'].append(avg_train_loss)
        train_history['train_accuracy'].append(train_accuracy)
        train_history['learning_rates'].append(optimizer.get_lr())

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

        train_history['val_loss'].append(avg_val_loss)
        train_history['val_accuracy'].append(val_accuracy)

        print(f'验证结果 - 损失: {avg_val_loss:.4f}, 准确率: {val_accuracy:.2f}%')

        # 保存最佳模型
        is_best = val_accuracy > best_accuracy
        if is_best:
            best_accuracy = val_accuracy

        # 保存检查点
        checkpoint_path = save_checkpoint(
            epoch, model, optimizer, scheduler, best_accuracy, train_history, is_best
        )

        if epoch % 5 == 0:
            print(f"💾 保存检查点: {checkpoint_path}")

        # 提前停止检查
        if epoch >= 10 and best_accuracy < 10.0:
            print("❌ 准确率过低，停止训练")
            break

        if epoch >= 15 and train_accuracy - val_accuracy > 40.0:
            print("❌ 过拟合严重，停止训练")
            break

    # 训练总结
    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print(f"🏆 最佳验证准确率: {best_accuracy:.2f}%")
    print("=" * 60)

    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_history['train_loss']) + 1),
        'train_loss': train_history['train_loss'],
        'train_accuracy': train_history['train_accuracy'],
        'val_loss': train_history['val_loss'],
        'val_accuracy': train_history['val_accuracy'],
        'learning_rate': train_history['learning_rates']
    })
    history_path = os.path.join(Config.save_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"📈 训练历史已保存: {history_path}")

    return model, history_df


if __name__ == "__main__":
    # 设置检查点路径
    Config.resume_checkpoint = '/home/aistudio/work/best_model.pdparams'
    model, history = train_improved_vit()