import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import time
import os
from collections import OrderedDict

# 设置CUDA内存分配策略
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# ==================== WideResNet模型定义 ====================
class WideBasicBlock(nn.Module):
    """WideResNet基础残差块"""

    def __init__(self, in_planes, out_planes, stride=1, dropout_rate=0.4):
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
        residual = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out += self.shortcut(residual)
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


# ==================== 实时识别类 ====================
class FlowerRecognizer:
    def __init__(self, model_path, class_names, device='cuda'):
        """
        初始化花卉识别器

        参数:
            model_path: 训练好的模型路径
            class_names: 类别名称列表
            device: 使用的设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载类别名称
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        print(f"加载了 {self.num_classes} 个花卉类别")

        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()  # 设置为评估模式

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self, path):
        """加载模型"""
        # 使用与训练时相同的参数
        model = WideResNet(
            depth=28,  # 与训练时保持一致
            widen_factor=4,  # 与训练时保持一致
            dropout_rate=0.0,  # 推理时不需要dropout
            num_classes=self.num_classes
        )

        # 加载模型权重
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # 检查是否是检查点文件（包含多个键）
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 从检查点文件中提取模型权重
            state_dict = checkpoint['model_state_dict']
        else:
            # 直接加载模型权重
            state_dict = checkpoint

        # 处理可能的DataParallel前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 去掉'module.'前缀
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model = model.to(self.device)

        return model

    def preprocess_frame(self, frame):
        """预处理摄像头帧"""
        # 转换BGR到RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 转换为PIL图像
        pil_image = Image.fromarray(frame_rgb)

        # 应用变换
        input_tensor = self.transform(pil_image)

        # 添加批次维度
        input_batch = input_tensor.unsqueeze(0)

        return input_batch.to(self.device)

    def predict(self, frame):
        """对帧进行预测"""
        # 预处理
        input_batch = self.preprocess_frame(frame)

        # 推理
        with torch.no_grad():
            output = self.model(input_batch)

        # 获取预测结果
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = self.class_names[predicted_idx.item()]

        return predicted_class, confidence.item()

    def run(self, camera_index=0):
        """运行实时识别"""
        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("无法打开摄像头")
            return

        print("按 'q' 键退出实时识别")
        print("按 'p' 键暂停/继续")

        # 设置帧率
        cap.set(cv2.CAP_PROP_FPS, 30)

        # 用于计算FPS
        prev_time = time.time()
        fps = 0

        # 暂停状态
        paused = False

        while True:
            if not paused:
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    print("无法读取帧")
                    break

                # 调整帧大小
                frame = cv2.resize(frame, (640, 480))

                # 进行预测
                predicted_class, confidence = self.predict(frame)

                # 计算FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

                # 在帧上绘制结果
                label = f"{predicted_class} ({confidence:.2f})"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit, 'p' to pause", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                # 显示帧
                cv2.imshow('Flower Recognition', frame)

            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                if paused:
                    print("已暂停")
                else:
                    print("继续")

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()


# ==================== 主函数 ====================
def main():
    # 设置路径
    model_path = "checkpoints/checkpoint_epoch_20.pth"  # 训练好的模型路径

    # 类别名称（根据您的实际类别修改）
    class_names = [
    'c0','c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24','c25','c26','c27','c28','c29','c30','c31','c32','c33','c34','c35','c36','c37','c38','c39','c40','c41','c42','c43','c44','c45','c46','c47','c48','c49','c50','c51','c52','c53','c54','c55','c56','c57','c58','c59','c60','c61','c62','c63','c64','c65','c66','c67','c68','c69','c70','c71','c72','c73','c74','c75','c76','c77','c78','c79','c80','c81','c82','c83','c84','c85','c86','c87','c88','c89','c90','c91','c92','c93','c94','c95','c96','c97','c98','c99','c100','c101'
    ]

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在")
        print("请先运行训练脚本并确保模型已保存")
        return

    # 创建识别器
    recognizer = FlowerRecognizer(model_path, class_names)

    # 运行实时识别
    recognizer.run(camera_index=0)  # 0表示默认摄像头


if __name__ == "__main__":
    main()
