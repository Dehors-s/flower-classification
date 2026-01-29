import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 改为True以提高性能
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed()

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")


# ==================== WideResNet模型定义 ====================
class WideBasicBlock(nn.Module):
    """WideResNet基础残差块"""

    def __init__(self, in_planes, out_planes, stride=1, dropout_rate=0.3):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # 短路连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=stride, bias=False)
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
    """WideResNet主网络"""

    def __init__(self, depth=16, widen_factor=2, dropout_rate=0.3, num_classes=102):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        # 计算每个块的层数
        assert (depth - 4) % 6 == 0, "深度应为6n+4"
        self.n = (depth - 4) // 6

        # 通道数配置
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # 三个残差块组
        self.layer1 = self._make_layer(WideBasicBlock, nChannels[0], nChannels[1],
                                       stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(WideBasicBlock, nChannels[1], nChannels[2],
                                       stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(WideBasicBlock, nChannels[2], nChannels[3],
                                       stride=2, dropout_rate=dropout_rate)

        # 全局平均池化和分类层
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
        layers = []
        layers.append(block(in_planes, out_planes, stride, dropout_rate))
        for _ in range(1, self.n):
            layers.append(block(out_planes, out_planes, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
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
def create_data_loaders(data_dir, batch_size=32):
    """创建数据加载器"""
    # 直接读取图像数据，只进行必要的转换
    data_transforms = transforms.Compose([
        transforms.ToTensor(),  # 只转换为Tensor
    ])

    # 创建数据集
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train_augmented'), data_transforms),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms)
    }

    # 创建数据加载器 - 优化参数
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,  # 增加工作进程数
            pin_memory=True,
            prefetch_factor=2,  # 预取数据
            persistent_workers=True  # 保持工作进程存活
        ),
        'val': torch.utils.data.DataLoader(
            image_datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            image_datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    print(f"训练集大小: {dataset_sizes['train']}")
    print(f"验证集大小: {dataset_sizes['val']}")
    print(f"测试集大小: {dataset_sizes['test']}")
    print(f"类别数量: {len(class_names)}")

    return dataloaders, dataset_sizes, class_names


# ==================== 训练函数 ====================
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=100,
                checkpoint_dir='checkpoints'):
    """模型训练函数"""
    since = time.time()

    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 初始化混合精度训练
    scaler = GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    # 记录训练历史
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # 外层进度条显示epoch进度
    epoch_pbar = tqdm(range(num_epochs), desc="总训练进度", unit="epoch")

    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()  # 评估模式

            running_loss = 0.0
            running_corrects = 0

            # 使用tqdm添加批次进度条
            dataloader = dataloaders[phase]
            batch_count = len(dataloader)
            batch_pbar = tqdm(total=batch_count, desc=f'{phase}批次', unit='batch', leave=False)

            # 迭代数据
            for inputs, labels in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        # 训练阶段使用混合精度
                        with autocast():
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                        # 反向传播+优化
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # 验证阶段不使用混合精度
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 更新批次进度条
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{torch.sum(preds == labels.data).double() / inputs.size(0):.4f}'
                })
                batch_pbar.update(1)

            batch_pbar.close()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # 记录历史
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.cpu().numpy())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.cpu().numpy())

            # 更新外层进度条信息
            if phase == 'val':
                epoch_pbar.set_postfix({
                    'Train_Loss': f'{train_loss_history[-1]:.4f}',
                    'Train_Acc': f'{train_acc_history[-1]:.4f}',
                    'Val_Loss': f'{val_loss_history[-1]:.4f}',
                    'Val_Acc': f'{val_acc_history[-1]:.4f}'
                })

            # 深拷贝最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())

        # 每20个epoch保存一次检查点
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_loss_history': train_loss_history,
                'train_acc_history': train_acc_history,
                'val_loss_history': val_loss_history,
                'val_acc_history': val_acc_history,
            }, checkpoint_path)
            print(f'\n检查点已保存: {checkpoint_path}')

        # 每个epoch结束后清空GPU缓存
        torch.cuda.empty_cache()

    epoch_pbar.close()

    time_elapsed = time.time() - since
    print(f'训练完成，用时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f} (第{best_epoch}轮)')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history


# ==================== 评估函数 ====================
def evaluate_model(model, dataloader, class_names):
    """模型评估函数"""
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    # 添加进度条
    pbar = tqdm(total=len(dataloader), desc='测试批次', unit='batch')

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'Acc': f'{correct / total:.4f}'
            })

    pbar.close()

    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')

    # 分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    return accuracy, all_preds, all_labels, all_probs


# ==================== 从检查点恢复训练 ====================
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """从检查点恢复训练"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    train_loss_history = checkpoint['train_loss_history']
    train_acc_history = checkpoint['train_acc_history']
    val_loss_history = checkpoint['val_loss_history']
    val_acc_history = checkpoint['val_acc_history']

    print(f"从检查点恢复训练，epoch {start_epoch}")
    return model, optimizer, scheduler, start_epoch, best_acc, train_loss_history, train_acc_history, val_loss_history, val_acc_history


# ==================== 主训练流程 ====================
def main():
    # 数据目录配置
    data_dir = './'  # 根据实际情况调整

    # 创建数据加载器
    print("加载数据...")
    dataloaders, dataset_sizes, class_names = create_data_loaders(data_dir, batch_size=16)

    # 初始化模型
    print("初始化WideResNet模型...")
    model = WideResNet(
        depth=16,  # 减小网络深度
        widen_factor=2,  # 减小宽度因子
        dropout_rate=0.3,  # Dropout率
        num_classes=len(class_names)
    )
    model = model.to(device)

    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 使用SGD优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,  # 初始学习率
        momentum=0.9,  # 动量
        weight_decay=5e-4,  # 权重衰减
        nesterov=True
    )

    # 学习率调度器（余弦退火）
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    # 如果需要从检查点恢复训练，取消下面的注释并指定检查点路径
    # checkpoint_path = 'checkpoints/checkpoint_epoch_20.pth'
    # model, optimizer, scheduler, start_epoch, best_acc, train_loss_history, train_acc_history, val_loss_history, val_acc_history = load_checkpoint(
    #     model, optimizer, scheduler, checkpoint_path)
    # 注意：如果从检查点恢复，需要修改train_model函数以接受这些参数

    # 训练模型
    print("开始训练...")
    model, train_loss, train_acc, val_loss, val_acc = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=100
    )

    # 保存最佳模型
    torch.save(model.state_dict(), 'best_flower_wrn_model.pth')
    print("模型已保存为 'best_flower_wrn_model.pth'")

    # 评估模型
    print("\n在测试集上评估模型...")
    test_accuracy, preds, labels, probs = evaluate_model(model, dataloaders['test'], class_names)

    # 绘制训练曲线
    plt.figure(figsize=(15, 5))

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
    plt.savefig('training_history.png')
    plt.show()

    print(f"最终测试准确率: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()