import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.io import DataLoader, Dataset
from paddle.optimizer import AdamW
from paddle.optimizer.lr import CosineAnnealingDecay, LinearWarmup
import os
import numpy as np
import time
import csv  # 使用csv替代pandas
from PIL import Image
import random
import math

print(f"PaddlePaddle版本: {paddle.__version__}")


# 优化的ViT配置 - 减小模型规模
class OptimizedViTConfig:
    data_dir = r'D:\ptcharm\project\花卉分析'

    # 两阶段训练策略
    image_size_stage1 = 224  # 第一阶段使用224
    image_size_stage2 = 384  # 第二阶段微调使用384

    patch_size = 16
    num_classes = 100
    batch_size_stage1 = 16  # 第一阶段可以用更大的batch
    batch_size_stage2 = 16  # 第二阶段减小batch

    # 减小模型规模以适应数据集
    dim = 384  # 减小嵌入维度
    depth = 6  # 减少层数
    heads = 8  # 保持注意力头
    mlp_ratio = 4
    dropout = 0.2  # 增加dropout防止过拟合

    # 学习率配置
    learning_rate_stage1 = 1e-4
    learning_rate_stage2 = 5e-5  # 微调时使用更小的学习率
    weight_decay = 1e-4
    warmup_epochs = 5
    num_epochs_stage1 = 50
    num_epochs_stage2 = 30

    save_dir = r'D:\ptcharm\project\花卉分析\output_vit_optimized'

    # 断点继续配置
    resume_stage1 = None  # 设置为具体路径如 '/home/aistudio/work/output_vit_optimized/checkpoint_stage1_epoch_20.pdparams'
    resume_stage2 = None  # 第二阶段恢复路径

    log_interval = 20

    os.makedirs(save_dir, exist_ok=True)


def setup_memory_optimization():
    """设置内存优化配置"""
    # 开启显存垃圾回收
    os.environ['FLAGS_eager_delete_tensor_gb'] = '0'
    os.environ['FLAGS_fast_eager_deletion_mode'] = 'True'
    os.environ['FLAGS_memory_fraction_of_eager_deletion'] = '1'

    # 调整显存分配策略
    os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.9'

    # 设置卷积工作空间大小限制
    os.environ['FLAGS_conv_workspace_size_limit'] = '512'


def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += int(param.numel())
    return total_params


class RandomGrayscale:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, paddle.Tensor):
                if img.shape[0] == 3:
                    gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                    img = paddle.stack([gray, gray, gray], axis=0)
            else:
                img = img.convert('L').convert('RGB')
        return img


class LabelSmoothingCrossEntropy(nn.Layer):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing

        logprobs = nn.functional.log_softmax(x, axis=-1)

        # 处理软标签和硬标签
        if len(target.shape) > 1 and target.shape[-1] == x.shape[-1]:
            # 软标签情况 (MixUp)
            targets = target
        else:
            # 硬标签情况
            targets = paddle.nn.functional.one_hot(
                target.astype('int64'),
                num_classes=x.shape[-1]
            )
            targets = targets * (1 - self.smoothing) + self.smoothing / x.shape[-1]

        # 更安全的损失计算
        nll_loss = - (targets * logprobs).sum(axis=-1)
        loss = nll_loss.mean()

        return loss


# 优化的ViT模型 - 添加更多正则化
class OptimizedViT(nn.Layer):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 dim=384, depth=6, heads=8, mlp_ratio=4, dropout=0.2):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Patch嵌入
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

        # 增强的分类头 - 添加dropout
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

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

        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose([0, 2, 1])

        cls_tokens = self.cls_token.expand([B, -1, -1])
        x = paddle.concat([cls_tokens, x], axis=1)

        x = x + self.pos_embedding
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)
        x = x[:, 0]
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
        residual = x
        x = self.norm1(x)
        attn_output = self.attn(x, x, x)
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.norm2(x)
        mlp_output = self.mlp(x)
        x = residual + self.dropout(mlp_output)

        return x


class FlowerDataset(Dataset):
    def __init__(self, data_dir, transform=None, use_mixup=False, alpha=0.2):
        self.data_dir = data_dir
        self.transform = transform
        self.use_mixup = use_mixup
        self.alpha = alpha
        self.samples = []
        self.labels = []

        categories = sorted([d for d in os.listdir(data_dir) if d.isdigit()], key=int)
        self.label_mapping = {int(cat): idx for idx, cat in enumerate(categories)}
        self.num_classes = len(categories)

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

            # 简单的MixUp数据增强 - 返回硬标签，避免形状不一致问题
            if self.use_mixup and self.transform and np.random.random() < 0.5:
                mixup_idx = np.random.randint(0, len(self.samples))
                mixup_img_path = self.samples[mixup_idx]
                mixup_label = self.labels[mixup_idx]

                mixup_img = Image.open(mixup_img_path).convert('RGB')
                mixup_img = self.transform(mixup_img)

                lam = np.random.beta(self.alpha, self.alpha)
                img = lam * img + (1 - lam) * mixup_img

                # 为了简化，我们返回硬标签而不是软标签
                # 这样可以避免数据加载器中的形状不一致问题
                if lam > 0.5:
                    # 返回原始标签
                    label = label
                else:
                    # 返回混合标签
                    label = mixup_label

            return img, label

        except Exception as e:
            print(f"加载图片失败: {img_path}, 错误: {e}")
            # 返回正确类型的占位符数据
            dummy_img = paddle.zeros([3, OptimizedViTConfig.image_size_stage1,
                                      OptimizedViTConfig.image_size_stage1]).astype('float32')
            dummy_label = 0
            return dummy_img, dummy_label

    def __len__(self):
        return len(self.samples)


def create_optimized_transforms(image_size=224, is_training=True):
    if is_training:
        train_transforms = T.Compose([
            T.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(10),
            # 修改hue参数为非负范围，避免出现负数导致的uint8溢出
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=(0, 0.1)),
            RandomGrayscale(p=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return train_transforms
    else:
        val_transforms = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return val_transforms


def create_model(image_size=224):
    model = OptimizedViT(
        image_size=image_size,
        patch_size=OptimizedViTConfig.patch_size,
        num_classes=OptimizedViTConfig.num_classes,
        dim=OptimizedViTConfig.dim,
        depth=OptimizedViTConfig.depth,
        heads=OptimizedViTConfig.heads,
        mlp_ratio=OptimizedViTConfig.mlp_ratio,
        dropout=OptimizedViTConfig.dropout
    )

    print(f"创建优化的ViT模型")
    print(f"输入尺寸: {image_size}")
    print(f"类别数: {OptimizedViTConfig.num_classes}")
    print(f"Patch数量: {(image_size // OptimizedViTConfig.patch_size) ** 2}")

    total_params = count_parameters(model)
    print(f"总参数量: {total_params:,}")

    return model


def create_optimizer_scheduler(model, total_steps, learning_rate, num_epochs, warmup_epochs):
    warmup_steps = warmup_epochs * total_steps // num_epochs

    cosine_scheduler = CosineAnnealingDecay(
        learning_rate=learning_rate,
        T_max=total_steps - warmup_steps
    )

    scheduler = LinearWarmup(
        learning_rate=cosine_scheduler,
        warmup_steps=warmup_steps,
        start_lr=learning_rate * 0.01,
        end_lr=learning_rate
    )

    # 创建梯度裁剪对象
    grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)

    # 修正：基于参数名称判断是否衰减，而不是参数对象
    no_decay = ['bias', 'LayerNorm.weight']
    # 收集不需要衰减的参数名称
    no_decay_names = [n for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)]

    # 优化器参数分组（基于名称）
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': OptimizedViTConfig.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]

    optimizer = AdamW(
        parameters=optimizer_grouped_parameters,
        learning_rate=scheduler,
        weight_decay=OptimizedViTConfig.weight_decay,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        grad_clip=grad_clip,
        # 修正：基于参数名称判断是否应用衰减（x是参数名称字符串）
        apply_decay_param_fun=lambda x: x not in no_decay_names
    )

    return optimizer, scheduler

def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_path=None):
    """加载检查点以恢复训练"""
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return 0, 0.0, [], [], [], []

    print(f"🔄 从检查点恢复训练: {checkpoint_path}")

    try:
        checkpoint = paddle.load(checkpoint_path)

        # 加载模型状态
        if 'model_state_dict' in checkpoint:
            model.set_state_dict(checkpoint['model_state_dict'])
            print("✅ 模型权重加载成功")
        else:
            model.set_state_dict(checkpoint)
            print("✅ 模型权重加载成功（简化格式）")

        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.set_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ 优化器状态加载成功")

        # 加载调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.set_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ 学习率调度器状态加载成功")

        # 加载训练状态
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)

        # 加载训练历史
        train_losses = checkpoint.get('train_losses', [])
        train_accuracies = checkpoint.get('train_accuracies', [])
        val_losses = checkpoint.get('val_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])

        print(f"📊 恢复训练状态: 从epoch {start_epoch}开始, 最佳准确率: {best_accuracy:.2f}%")

        return start_epoch, best_accuracy, train_losses, train_accuracies, val_losses, val_accuracies

    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        print("🔄 将从头开始训练")
        return 0, 0.0, [], [], [], []


def save_checkpoint(epoch, model, optimizer, scheduler, best_accuracy,
                    train_losses, train_accuracies, val_losses, val_accuracies,
                    stage=1, is_best=False):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_accuracy': best_accuracy,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'stage': stage,
        'config': {
            'image_size': OptimizedViTConfig.image_size_stage1 if stage == 1 else OptimizedViTConfig.image_size_stage2,
            'dim': OptimizedViTConfig.dim,
            'depth': OptimizedViTConfig.depth,
            'heads': OptimizedViTConfig.heads,
            'learning_rate': OptimizedViTConfig.learning_rate_stage1 if stage == 1 else OptimizedViTConfig.learning_rate_stage2,
        }
    }

    if is_best:
        checkpoint_path = os.path.join(OptimizedViTConfig.save_dir, f'best_model_stage{stage}.pdparams')
    else:
        checkpoint_path = os.path.join(OptimizedViTConfig.save_dir, f'checkpoint_stage{stage}_epoch_{epoch}.pdparams')

    paddle.save(checkpoint, checkpoint_path)
    return checkpoint_path


def save_training_history(train_losses, train_accuracies, val_losses, val_accuracies, stage=1):
    """使用csv保存训练历史"""
    history_path = os.path.join(OptimizedViTConfig.save_dir, f'training_history_stage{stage}.csv')

    with open(history_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])

        for epoch, (train_loss, train_acc, val_loss, val_acc) in enumerate(
                zip(train_losses, train_accuracies, val_losses, val_accuracies), 1
        ):
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

    print(f"📈 训练历史已保存: {history_path}")


def train_stage(stage=1, resume_checkpoint=None):
    """两阶段训练 - 带断点继续功能"""
    if stage == 1:
        print("=" * 60)
        print("第一阶段：224尺寸训练")
        print("=" * 60)
        image_size = OptimizedViTConfig.image_size_stage1
        batch_size = OptimizedViTConfig.batch_size_stage1
        num_epochs = OptimizedViTConfig.num_epochs_stage1
        learning_rate = OptimizedViTConfig.learning_rate_stage1
        use_mixup = False  # 暂时关闭MixUp，避免数据加载问题
        # 使用配置中的恢复路径，如果没有则使用传入的参数
        resume_path = OptimizedViTConfig.resume_stage1 if OptimizedViTConfig.resume_stage1 else resume_checkpoint
    else:
        print("=" * 60)
        print("第二阶段：384尺寸微调")
        print("=" * 60)
        image_size = OptimizedViTConfig.image_size_stage2
        batch_size = OptimizedViTConfig.batch_size_stage2
        num_epochs = OptimizedViTConfig.num_epochs_stage2
        learning_rate = OptimizedViTConfig.learning_rate_stage2
        use_mixup = False  # 微调时关闭MixUp
        # 使用配置中的恢复路径，如果没有则使用传入的参数
        resume_path = OptimizedViTConfig.resume_stage2 if OptimizedViTConfig.resume_stage2 else resume_checkpoint

    train_dir = os.path.join(OptimizedViTConfig.data_dir, 'train')
    val_dir = os.path.join(OptimizedViTConfig.data_dir, 'val')

    train_transforms = create_optimized_transforms(image_size, is_training=True)
    val_transforms = create_optimized_transforms(image_size, is_training=False)

    print("加载训练集...")
    train_dataset = FlowerDataset(train_dir, transform=train_transforms, use_mixup=use_mixup)
    print("加载验证集...")
    val_dataset = FlowerDataset(val_dir, transform=val_transforms, use_mixup=False)

    # 使用单进程数据加载，避免多进程问题
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # 设置为0避免多进程问题
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # 设置为0避免多进程问题
    )

    print(f"\n阶段{stage}配置:")
    print(f"图像尺寸: {image_size}")
    print(f"批次大小: {batch_size}")
    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")

    # 创建模型
    model = create_model(image_size)

    total_steps = len(train_loader) * num_epochs
    optimizer, scheduler = create_optimizer_scheduler(
        model, total_steps, learning_rate, num_epochs, OptimizedViTConfig.warmup_epochs
    )

    # 初始化训练状态
    start_epoch = 0
    best_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # 加载检查点（如果提供）
    if resume_path:
        start_epoch, best_accuracy, train_losses, train_accuracies, val_losses, val_accuracies = load_checkpoint(
            model, optimizer, scheduler, resume_path
        )

    # 使用标签平滑损失
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    patience = 10  # 增加耐心值
    patience_counter = 0

    print(f"\n开始阶段{stage}训练从epoch {start_epoch + 1}到{num_epochs}...")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):
        current_epoch = epoch + 1
        print(f"\nEpoch {current_epoch}/{num_epochs}")
        print("-" * 50)

        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.clear_grad()

            # 确保数据在正确的设备上
            data = data.astype('float32')
            target = target.astype('int64')  # 确保标签是int64类型

            output = model(data)
            loss = criterion(output, target)

            loss.backward()

            # 注意：梯度裁剪已在优化器中设置，不需要单独调用
            # 删除 paddle.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            pred = output.argmax(axis=1)
            correct += (pred == target).sum().item()
            total += target.shape[0]

            if batch_idx % OptimizedViTConfig.log_interval == 0:
                current_lr = optimizer.get_lr()
                batch_acc = 100. * correct / total if total > 0 else 0
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
                data = data.astype('float32')
                target = target.astype('int64')

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

        # 检查是否是最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                current_epoch, model, optimizer, scheduler, best_accuracy,
                train_losses, train_accuracies, val_losses, val_accuracies,
                stage=stage, is_best=True
            )
            patience_counter = 0
            print(f"🚀 保存最佳模型，准确率: {val_accuracy:.2f}%")
        else:
            patience_counter += 1
            print(f"⏳ 早停计数: {patience_counter}/{patience}")

        # 每5个epoch保存一次检查点
        if current_epoch % 5 == 0:
            checkpoint_path = save_checkpoint(
                current_epoch, model, optimizer, scheduler, best_accuracy,
                train_losses, train_accuracies, val_losses, val_accuracies,
                stage=stage, is_best=False
            )
            print(f"💾 保存检查点: {checkpoint_path}")

        # 早停检查
        if patience_counter >= patience:
            print(f"🛑 早停触发，在epoch {current_epoch}停止训练")
            break

    # 保存最终模型
    final_checkpoint_path = save_checkpoint(
        num_epochs, model, optimizer, scheduler, best_accuracy,
        train_losses, train_accuracies, val_losses, val_accuracies,
        stage=stage, is_best=False
    )
    print(f"💾 保存最终模型: {final_checkpoint_path}")

    # 保存训练历史到CSV
    save_training_history(train_losses, train_accuracies, val_losses, val_accuracies, stage)

    print(f"\n🎉 阶段{stage}训练完成!")
    print(f"🏆 最佳验证准确率: {best_accuracy:.2f}%")

    return model, best_accuracy


def main():
    """修复后的主训练函数"""
    # 设置内存优化
    setup_memory_optimization()

    # 设置随机种子
    paddle.seed(42)
    np.random.seed(42)
    random.seed(42)

    print("开始第一阶段训练...")
    try:
        model_stage1, best_acc_stage1 = train_stage(stage=1)

        if best_acc_stage1 > 45:
            print("\n开始第二阶段微调...")
            resume_path = os.path.join(OptimizedViTConfig.save_dir, 'best_model_stage1.pdparams')
            model_stage2, best_acc_stage2 = train_stage(stage=2, resume_checkpoint=resume_path)

            print(f"\n训练总结:")
            print(f"第一阶段(224)最佳准确率: {best_acc_stage1:.2f}%")
            print(f"第二阶段(384)最佳准确率: {best_acc_stage2:.2f}%")
            print(f"提升: {best_acc_stage2 - best_acc_stage1:+.2f}%")
        else:
            print(f"\n第一阶段准确率({best_acc_stage1:.2f}%)不足，跳过第二阶段微调")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()