# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# 在导入部分添加安全全局变量
import os
import numpy
import torch.serialization
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct, numpy.ndarray])
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


# 检测设备
def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = f"使用设备: {device}"

    if torch.cuda.is_available():
        device_info += f"\nGPU名称: {torch.cuda.get_device_name(0)}"
        device_info += f"\nGPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB"

    return device, device_info


# WideResNet模型定义（与训练代码保持一致）
class WideBasicBlock(nn.Module):
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
    def __init__(self, depth=16, widen_factor=2, dropout_rate=0.3, num_classes=102):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "深度应为6n+4"
        self.n = (depth - 4) // 6

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(WideBasicBlock, nChannels[0], nChannels[1],
                                       stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(WideBasicBlock, nChannels[1], nChannels[2],
                                       stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(WideBasicBlock, nChannels[2], nChannels[3],
                                       stride=2, dropout_rate=dropout_rate)

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nChannels[3], num_classes)

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


# 数据加载器
def create_data_loaders(data_dir, batch_size=32):
    """创建数据加载器"""
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_datasets = {
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms)
    }

    dataloaders = {
        'test': torch.utils.data.DataLoader(
            image_datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    }

    dataset_sizes = {'test': len(image_datasets['test'])}
    class_names = image_datasets['test'].classes

    print(f"测试集大小: {dataset_sizes['test']}")
    print(f"类别数量: {len(class_names)}")

    return dataloaders, dataset_sizes, class_names


# 模型评估函数
def evaluate_model(model, dataloader, class_names, device):
    """模型评估函数"""
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

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
    plt.savefig('confusion_matrix_checkpoint.png')
    plt.show()

    return accuracy, all_preds, all_labels, all_probs


# 加载检查点并评估
def evaluate_checkpoint(checkpoint_path, data_dir, num_classes=None):
    """评估检查点文件"""
    # 设置随机种子
    set_seed()

    # 检测设备
    device, device_info = setup_device()
    print(device_info)

    # 创建数据加载器
    print("加载数据...")
    dataloaders, dataset_sizes, class_names = create_data_loaders(data_dir, batch_size=16)

    # 如果未指定类别数量，使用数据中的类别数量
    if num_classes is None:
        num_classes = len(class_names)

    # 初始化模型
    print("初始化模型...")
    model = WideResNet(
        depth=16,
        widen_factor=2,
        dropout_rate=0.3,
        num_classes=num_classes
    )
    model = model.to(device)

    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 加载检查点
    print(f"加载检查点: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 检查检查点内容并适当加载
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint

    # 处理可能的键名不匹配（如多GPU训练保存的模型）
    if list(model_state_dict.keys())[0].startswith('module.'):
        # 移除'module.'前缀（多GPU训练保存的模型）
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model_state_dict = new_state_dict

    # 加载模型权重
    model.load_state_dict(model_state_dict)

    # 评估模型
    print("\n在测试集上评估模型...")
    test_accuracy, preds, labels, probs = evaluate_model(model, dataloaders['test'], class_names, device)

    # 返回评估结果
    return {
        'accuracy': test_accuracy,
        'predictions': preds,
        'labels': labels,
        'probabilities': probs,
        'class_names': class_names
    }


# 主函数
if __name__ == "__main__":
    # 检查点路径
    checkpoint_path = "checkpoints/checkpoint_epoch_20.pth"

    # 数据目录
    data_dir = "./"

    try:
        # 评估检查点
        results = evaluate_checkpoint(checkpoint_path, data_dir)

        # 打印最终结果
        print(f"\n最终测试准确率: {results['accuracy']:.2f}%")

        # 保存评估结果
        with open('checkpoint_evaluation_results.txt', 'w') as f:
            f.write(f"检查点文件: {checkpoint_path}\n")
            f.write(f"测试准确率: {results['accuracy']:.2f}%\n")
            f.write(f"类别数量: {len(results['class_names'])}\n")
            f.write(f"测试样本数: {len(results['labels'])}\n")

        print("评估完成，结果已保存到 'checkpoint_evaluation_results.txt'")

    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        import traceback

        traceback.print_exc()