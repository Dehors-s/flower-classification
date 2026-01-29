import torch

import paddle

import numpy as np
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))
# 添加models目录到路径以导入vit_v1
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, 'models')
sys.path.append(models_dir)

from vit_v1 import create_paddle_model, Config


def compare_models():
    """对比PaddlePaddle和PyTorch模型"""
    print("=" * 60)
    print("模型参数对比 - 修复版")
    print("=" * 60)

    # 创建PaddlePaddle模型
    paddle_model = create_paddle_model()
    paddle_model.eval()

    print(f"PaddlePaddle版本: {paddle.__version__}")
    print(f"PyTorch版本: {torch.__version__}")
    print()

    # 统计参数
    def count_parameters(model):
        total_params = 0
        for param in model.parameters():
            total_params += int(param.numel())
        return total_params

    paddle_params = count_parameters(paddle_model)
    print(f"Paddle模型参数量: {paddle_params:,}")

    # 测试前向传播
    print("\n测试Paddle模型前向传播...")
    test_input = paddle.randn([2, 3, Config.image_size, Config.image_size])

    with paddle.no_grad():
        paddle_output = paddle_model(test_input)

    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {paddle_output.shape}")
    print(f"输出范围: [{paddle_output.min().item():.3f}, {paddle_output.max().item():.3f}]")

    # 测试梯度
    print("\n测试Paddle模型梯度...")
    paddle_model.train()
    criterion = paddle.nn.CrossEntropyLoss()
    test_target = paddle.randint(0, Config.num_classes, [2])

    paddle_output = paddle_model(test_input)
    loss = criterion(paddle_output, test_target)
    loss.backward()

    # 检查梯度
    grad_norms = []
    for name, param in paddle_model.named_parameters():
        if param.grad is not None:
            grad_norm = paddle.norm(param.grad).item()
            grad_norms.append(grad_norm)
            if len(grad_norms) <= 5:  # 只显示前5个
                print(f"  {name}: 梯度范数 = {grad_norm:.6f}")

    if grad_norms:
        print(f"梯度范数范围: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
        print("✅ Paddle模型梯度正常")
    else:
        print("❌ 没有检测到梯度")

    print("\n" + "=" * 60)
    print("对比完成!")
    print("=" * 60)

    return paddle_model


if __name__ == "__main__":
    model = compare_models()