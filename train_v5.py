import os
# 设置环境变量减少内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import time
import copy
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed()


# 配置参数 - 内存优化版本
class Config:
    data_dir = r'D:\ptcharm\project\花卉分析'
    num_classes = 100
    image_size = 224
    batch_size = 16  # 减小批次大小
    accumulation_steps = 2  # 梯度累积
    learning_rate = 0.1
    weight_decay = 5e-4
    num_epochs = 100
    save_dir = './output_wrn28_optimized'

    # WRN配置
    wrn_depth = 28
    wrn_width = 8
    dropout_rate = 0.3
    use_checkpoint = True  # 启用梯度检查点

    os.makedirs(save_dir, exist_ok=True)


# ==================== WideResNet模型（带梯度检查点） ====================
class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dropout_rate=0.3):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=8, dropout_rate=0.3, num_classes=100, use_checkpoint=True):
        super(WideResNet, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.in_planes = 16
        self.n = (depth - 4) // 6
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(WideBasicBlock, nChannels[0], nChannels[1], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(WideBasicBlock, nChannels[1], nChannels[2], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(WideBasicBlock, nChannels[2], nChannels[3], stride=2, dropout_rate=dropout_rate)

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nChannels[3], num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_planes, out_planes, stride, dropout_rate):
        layers = [block(in_planes, out_planes, stride, dropout_rate)]
        for _ in range(1, self.n):
            layers.append(block(out_planes, out_planes, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        # 使用梯度检查点（仅在训练时）
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            out = checkpoint(self.layer1, out)
            out = checkpoint(self.layer2, out)
            out = checkpoint(self.layer3, out)
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)

        out = self.bn1(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ==================== 数据加载 ====================
class FlowerDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []

        categories = sorted([d for d in os.listdir(data_dir) if d.isdigit()], key=int)
        self.label_mapping = {int(cat): idx for idx, cat in enumerate(categories)}

        print(f"加载数据集: {data_dir}")
        print(f"发现类别: {len(categories)}个")

        for category in categories:
            category_dir = os.path.join(data_dir, category)
            if os.path.isdir(category_dir):
                image_files = [f for f in os.listdir(category_dir)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for file in image_files:
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
            dummy_img = torch.rand(3, Config.image_size, Config.image_size)
            return dummy_img, label

    def __len__(self):
        return len(self.samples)


def create_data_loaders():
    """创建数据加载器"""
    # WRN需要更强的数据增强
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(Config.image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FlowerDataset(os.path.join(Config.data_dir, 'train'), transform=train_transforms)
    val_dataset = FlowerDataset(os.path.join(Config.data_dir, 'val'), transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    print(f"训练集大小: {dataset_sizes['train']}")
    print(f"验证集大小: {dataset_sizes['val']}")

    return {'train': train_loader, 'val': val_loader}, dataset_sizes


# ==================== 内存监控函数 ====================
def print_memory_usage(desc=""):
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"{desc}: 已分配: {allocated:.2f}GB, 保留: {reserved:.2f}GB, 峰值: {max_allocated:.2f}GB")


# ==================== 训练函数（修复混合精度问题） ====================
def train_model_wrn(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=100):
    """WRN专用训练函数（内存优化版）"""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 添加GradScaler用于混合精度训练（兼容旧版本）
    scaler = torch.cuda.amp.GradScaler() if hasattr(torch.cuda.amp, 'GradScaler') else None

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    print("开始训练WRN（内存优化版）...")
    print_memory_usage("训练开始前")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 40)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            start_time = time.time()

            # 在验证阶段使用no_grad减少内存使用
            with torch.set_grad_enabled(phase == 'train'):
                for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    if phase == 'train':
                        # 使用混合精度前向传播（兼容不同PyTorch版本）
                        if scaler is not None:
                            # 新版本PyTorch的混合精度
                            with torch.cuda.amp.autocast():
                                outputs = model(inputs)
                                _, preds = torch.max(outputs, 1)
                                loss = criterion(outputs, labels)
                                loss = loss / Config.accumulation_steps  # 梯度累积

                            # 使用scaler进行反向传播
                            scaler.scale(loss).backward()

                            # 只在累积步骤达到时更新权重
                            if (batch_idx + 1) % Config.accumulation_steps == 0:
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                        else:
                            # 旧版本PyTorch，不使用混合精度
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            loss = loss / Config.accumulation_steps

                            loss.backward()
                            if (batch_idx + 1) % Config.accumulation_steps == 0:
                                optimizer.step()
                                optimizer.zero_grad()
                    else:
                        # 验证阶段不需要混合精度
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0) * (
                        Config.accumulation_steps if phase == 'train' else 1)
                    running_corrects += torch.sum(preds == labels.data)

                    # 每50个batch打印一次内存使用情况
                    if batch_idx % 50 == 0:
                        print_memory_usage(f"Batch {batch_idx}")

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            epoch_time = time.time() - start_time
            print(f'{phase:5} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {epoch_time:.1f}s')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.cpu().numpy())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.cpu().numpy())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"🎉 新的最佳准确率: {best_acc:.4f}")

                # 立即保存最佳模型
                best_model_path = os.path.join(Config.save_dir, f'best_model_epoch_{epoch + 1}_acc_{best_acc:.4f}.pth')
                torch.save(best_model_wts, best_model_path)

        # 每20个epoch保存检查点
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(Config.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, checkpoint_path)
            print(f'检查点已保存: {checkpoint_path}')

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    time_elapsed = time.time() - since
    print(f'训练完成，用时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history


# ==================== 主训练流程 ====================
def main():
    print("=== WRN-28内存优化训练模式 ===")

    # 创建数据加载器
    print("加载数据...")
    dataloaders, dataset_sizes = create_data_loaders()

    # 初始化WRN-28模型（带梯度检查点）
    print("初始化WRN-28模型（带梯度检查点）...")
    model = WideResNet(
        depth=Config.wrn_depth,
        widen_factor=Config.wrn_width,
        dropout_rate=Config.dropout_rate,
        num_classes=Config.num_classes,
        use_checkpoint=Config.use_checkpoint
    )
    model = model.to(device)

    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # 定义损失函数 - 使用标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # WRN使用SGD + Momentum
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.learning_rate,
        momentum=0.9,
        weight_decay=Config.weight_decay,
        nesterov=True
    )

    # 学习率调度 - WRN使用阶梯下降
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # 训练模型
    model, train_loss, train_acc, val_loss, val_acc = train_model_wrn(
        model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, Config.num_epochs
    )

    # 保存最终模型
    final_model_path = os.path.join(Config.save_dir, 'final_wrn28_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存: '{final_model_path}'")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='训练损失')
    plt.plot(val_loss, label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='训练准确率')
    plt.plot(val_acc, label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_dir, 'training_history_wrn28.png'), dpi=150)
    plt.show()

    # 保存训练历史
    history = pd.DataFrame({
        'epoch': range(1, len(train_loss) + 1),
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc
    })
    history_path = os.path.join(Config.save_dir, 'training_history_wrn28.csv')
    history.to_csv(history_path, index=False)
    print(f"训练历史已保存: {history_path}")

    print(f"\n最终训练准确率: {train_acc[-1]:.4f}")
    print(f"最终验证准确率: {val_acc[-1]:.4f}")


if __name__ == "__main__":
    main()