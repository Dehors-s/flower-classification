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
from PIL import Image
import glob

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


# 数据加载器 - 用于目录形式的测试集
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


# 加载单张或多张图片 - 支持手动输入类别名
def load_single_or_multiple_images(image_path, class_names, batch_size=16):
    """
    加载单张图片或多张图片，支持手动输入每张图片的类别名

    参数:
        image_path: 可以是单张图片路径或包含图片的目录
        class_names: 所有类别的名称列表
        batch_size: 批处理大小

    返回:
        dataloader: 数据加载器
        labels: 真实标签列表（手动输入）
        num_samples: 样本数量
    """
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    images = []
    labels = []  # 存储手动输入的标签索引

    # 检查路径类型
    if os.path.isfile(image_path):
        # 单张图片
        image_paths = [image_path]
    elif os.path.isdir(image_path):
        # 目录中的所有图片
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(image_path, ext)))
        image_paths = sorted(image_paths)
    else:
        raise ValueError(f"无效的图片路径: {image_path}")

    if not image_paths:
        raise ValueError(f"未找到图片: {image_path}")

    # 显示可用类别，方便用户输入
    print("\n可用类别列表:")
    for i, cls in enumerate(class_names):
        print(f"{i}. {cls}")
    print()

    # 加载图片并手动输入标签
    for idx, img_path in enumerate(image_paths, 1):
        # 打开图片
        try:
            img = Image.open(img_path).convert('RGB')
            img = data_transforms(img)
            images.append(img)
        except Exception as e:
            print(f"警告: 无法加载图片 {img_path}，错误: {e}")
            continue

        # 手动输入类别名
        while True:
            print(f"处理第 {idx}/{len(image_paths)} 张图片: {os.path.basename(img_path)}")
            class_name = input("请输入该图片的类别名（或输入序号）: ").strip()

            # 检查是否输入的是序号
            if class_name.isdigit():
                class_idx = int(class_name)
                if 0 <= class_idx < len(class_names):
                    class_name = class_names[class_idx]
                    break
                else:
                    print(f"无效序号！请输入0到{len(class_names) - 1}之间的序号")
            # 检查类别名是否有效
            elif class_name in class_names:
                break
            else:
                print(f"无效类别名！请从以下类别中选择: {', '.join(class_names)}")

        # 获取类别索引
        labels.append(class_names.index(class_name))
        print(f"已确认类别: {class_name} (索引: {labels[-1]})\n")

    if not images:
        raise ValueError("没有成功加载任何图片")

    # 创建数据集和数据加载器
    dataset = torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return dataloader, labels, len(images)


# 模型评估函数
def evaluate_model(model, dataloader, class_names, device, true_labels=None):
    """模型评估函数，支持外部提供真实标签"""
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(total=len(dataloader), desc='测试批次', unit='batch')

    with torch.no_grad():
        for batch in dataloader:
            # 处理不同类型的数据加载器，确保inputs和labels是Tensor
            if isinstance(batch, tuple) and len(batch) == 2:
                inputs, labels = batch
                # 确保inputs是Tensor
                inputs = torch.as_tensor(inputs)
                if true_labels is None:
                    true_labels_batch = torch.as_tensor(labels)
                else:
                    true_labels_batch = torch.as_tensor(true_labels[:len(inputs)])
            else:
                # 直接处理batch，确保转换为Tensor
                inputs = torch.as_tensor(batch)
                true_labels_batch = torch.as_tensor(true_labels[:len(inputs)])

            # 后续设备转换代码（保持不变）
            inputs = inputs.to(device, non_blocking=True)
            true_labels_batch = true_labels_batch.to(device, non_blocking=True)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            total += true_labels_batch.size(0)
            correct += (preds == true_labels_batch).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(true_labels_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.update(1)
            pbar.set_postfix({
                'Acc': f'{correct / total:.4f}'
            })

    pbar.close()

    accuracy = 100 * correct / total
    print(f'识别准确率: {accuracy:.2f}%')

    # 分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 计算每个类别的识别率
    cm = confusion_matrix(all_labels, all_preds)
    class_accuracies = []

    for i in range(len(class_names)):
        # 每个类别的识别率 = 正确预测的数量 / 该类别的总样本数
        if cm[i].sum() > 0:  # 避免除以零
            acc = 100 * cm[i, i] / cm[i].sum()
            class_accuracies.append(acc)
            print(f"类别 '{class_names[i]}' 的识别率: {acc:.2f}%")
        else:
            class_accuracies.append(0.0)
            print(f"类别 '{class_names[i]}' 没有测试样本")

    # 混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig('confusion_matrix_checkpoint.png')
    plt.show()

    return accuracy, all_preds, all_labels, all_probs, class_accuracies


# 加载检查点并评估
def evaluate_checkpoint(checkpoint_path, data_source, num_classes=None, mode='directory', batch_size=16):
    """
    评估检查点文件

    参数:
        checkpoint_path: 模型检查点路径
        data_source: 数据来源，可以是目录、单张图片路径或包含图片的目录
        num_classes: 类别数量
        mode: 模式，'directory'表示传统测试集目录，'image'表示单张或多张图片
        batch_size: 批处理大小
    """
    # 设置随机种子
    set_seed()

    # 检测设备
    device, device_info = setup_device()
    print(device_info)

    # 根据模式加载数据
    print("加载数据...")
    if mode == 'directory':
        # 传统测试集目录模式
        dataloaders, dataset_sizes, class_names = create_data_loaders(data_source, batch_size=batch_size)
        dataloader = dataloaders['test']
        true_labels = None
        num_samples = dataset_sizes['test']
    else:
        # 单张/多张图片模式
        # 先获取类别名称（从训练时的类别获取，这里假设与测试集目录结构一致）
        # 尝试从标准测试集目录获取类别名称
        try:
            temp_dataloaders, _, class_names = create_data_loaders(
                os.path.dirname(data_source) if os.path.isfile(data_source) else data_source)
        except:
            # 如果无法获取，使用检查点中的信息或提示用户
            if num_classes is None:
                raise ValueError("无法自动获取类别名称，请提供num_classes参数")
            class_names = [f"class_{i}" for i in range(num_classes)]

        dataloader, true_labels, num_samples = load_single_or_multiple_images(data_source, class_names,
                                                                              batch_size=batch_size)
        true_labels = torch.tensor(true_labels)

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
    print("\n评估模型...")
    test_accuracy, preds, labels, probs, class_accuracies = evaluate_model(
        model, dataloader, class_names, device, true_labels)

    # 返回评估结果
    return {
        'accuracy': test_accuracy,
        'class_accuracies': class_accuracies,
        'predictions': preds,
        'labels': labels,
        'probabilities': probs,
        'class_names': class_names,
        'num_samples': num_samples
    }


# 主函数 - 统一管理所有输入变量
if __name__ == "__main__":
    # 统一配置所有输入变量
    config = {
        'checkpoint_path': "checkpoints/checkpoint_epoch_20.pth",  # 模型检查点路径
        'data_source': "test_augmented/c0/image_06734.jpg",  # 数据来源：目录/单张图片/图片目录
        'mode': 'image',  # 模式：'directory'（测试集目录）或 'image'（图片/图片目录）
        'num_classes': 102,  # 类别数量，None则自动获取
        'batch_size': 16,  # 批处理大小
        'result_save_path': 'checkpoint_evaluation_results.txt'  # 评估结果保存路径
    }

    try:
        # 评估检查点
        results = evaluate_checkpoint(
            checkpoint_path=config['checkpoint_path'],
            data_source=config['data_source'],
            num_classes=config['num_classes'],
            mode=config['mode'],
            batch_size=config['batch_size']
        )

        # 打印最终结果
        print(f"\n最终识别准确率: {results['accuracy']:.2f}%")

        # 保存评估结果
        with open(config['result_save_path'], 'w') as f:
            f.write(f"检查点文件: {config['checkpoint_path']}\n")
            f.write(f"数据来源: {config['data_source']}\n")
            f.write(f"识别准确率: {results['accuracy']:.2f}%\n")
            f.write(f"类别数量: {len(results['class_names'])}\n")
            f.write(f"测试样本数: {results['num_samples']}\n\n")
            f.write("每个类别的识别率:\n")
            for class_name, acc in zip(results['class_names'], results['class_accuracies']):
                f.write(f"  {class_name}: {acc:.2f}%\n")

        print(f"评估完成，结果已保存到 '{config['result_save_path']}'")

    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        import traceback

        traceback.print_exc()