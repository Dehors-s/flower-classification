import torch
import paddle
import torch.nn as nn
import numpy as np
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from vit_v1 import create_paddle_model, Config

# ==================== 配置区域 ====================
# 直接指定PaddlePaddle模型路径
PADDLE_MODEL_PATH = r"D:\ptcharm\project\花卉分析\checkpoint_epoch_30.pdparams"
# 指定输出的PyTorch模型文件名
PYTORCH_MODEL_NAME = "pytorch_vit_epoch_30.pth"
# ================================================

print(f"正在查找模型文件: {PADDLE_MODEL_PATH}")


# PyTorch版本的ViT模型定义（与PaddlePaddle版本结构相同）
class PyTorchViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 dim=384, depth=6, heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

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
            PyTorchTransformerBlock(dim, heads, mlp_dim, dropout)
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
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        # 使用卷积进行patch嵌入
        x = self.patch_embed(x)  # [B, dim, H//P, W//P]
        x = x.flatten(2)  # [B, dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, dim]

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
        x = x[:, 0]  # 取类别token
        x = self.head(x)

        return x


class PyTorchTransformerBlock(nn.Module):
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


def convert_paddle_to_pytorch(paddle_model_path=None):
    """将PaddlePaddle模型转换为PyTorch模型"""
    print("=" * 60)
    print("开始转换PaddlePaddle模型到PyTorch")
    print("=" * 60)

    # 如果未提供路径，使用配置的路径
    if paddle_model_path is None:
        paddle_model_path = PADDLE_MODEL_PATH

    print(f"查找模型文件: {paddle_model_path}")

    if not os.path.exists(paddle_model_path):
        # 尝试在常见位置查找模型文件
        possible_paths = [
            paddle_model_path,
            os.path.join(os.path.dirname(__file__), 'best_model.pdparams'),
            os.path.join(os.path.dirname(__file__), 'best_model2.pdparams'),
            os.path.join(os.path.dirname(__file__), 'output_vit_fixed', 'best_model.pdparams'),
            'best_model.pdparams',
            'best_model2.pdparams',
            os.path.expanduser('~/best_model.pdparams'),
        ]

        found = False
        for path in possible_paths:
            if os.path.exists(path):
                paddle_model_path = path
                found = True
                print(f"✅ 找到模型文件: {paddle_model_path}")
                break

        if not found:
            print("❌ 找不到训练好的模型文件")
            print("请提供正确的模型路径或先训练模型")
            return None

    # 创建PaddlePaddle模型并加载权重
    paddle_model = create_paddle_model()

    try:
        # 加载PaddlePaddle权重
        paddle_state_dict = paddle.load(paddle_model_path)
        paddle_model.set_state_dict(paddle_state_dict)
        paddle_model.eval()
        print(f"✅ 已加载PaddlePaddle权重: {paddle_model_path}")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None

    # 创建PyTorch模型
    pytorch_model = PyTorchViT(
        image_size=Config.image_size,
        patch_size=16,
        num_classes=Config.num_classes,
        dim=384,
        depth=6,
        heads=8,
        mlp_ratio=4,
        dropout=0.1
    )
    pytorch_model.eval()
    print("✅ 已创建PyTorch模型")

    # 转换权重
    pytorch_state_dict = {}

    for paddle_key, paddle_param in paddle_model.state_dict().items():
        # 转换参数格式
        param_np = paddle_param.numpy()

        # 处理不同形状的参数
        if paddle_param.ndim == 4:  # 卷积权重 [out_c, in_c, h, w]
            param_torch = torch.from_numpy(param_np)
            pytorch_key = paddle_key

        elif paddle_param.ndim == 2:  # 线性权重
            # PaddlePaddle: [out_features, in_features]
            # PyTorch: [in_features, out_features]
            # 需要转置
            param_torch = torch.from_numpy(param_np).T
            pytorch_key = paddle_key

        elif paddle_param.ndim == 1:  # 偏置或归一化参数
            param_torch = torch.from_numpy(param_np)
            pytorch_key = paddle_key

        elif paddle_param.ndim == 3:  # cls_token 和 pos_embedding
            param_torch = torch.from_numpy(param_np)
            pytorch_key = paddle_key

        else:
            print(f"⚠️  未知维度: {paddle_key} - {paddle_param.shape}")
            continue

        pytorch_state_dict[pytorch_key] = param_torch
        print(f"✅ 转换: {paddle_key} {paddle_param.shape} -> {pytorch_key} {param_torch.shape}")

    # 特殊处理多头注意力权重
    print("\n处理多头注意力权重...")

    # 对于每个Transformer层，处理多头注意力
    for i in range(6):  # depth=6
        # 获取Q、K、V权重和偏置
        q_weight = paddle_model.state_dict()[f'encoder_layers.{i}.attn.q_proj.weight'].numpy()
        k_weight = paddle_model.state_dict()[f'encoder_layers.{i}.attn.k_proj.weight'].numpy()
        v_weight = paddle_model.state_dict()[f'encoder_layers.{i}.attn.v_proj.weight'].numpy()

        q_bias = paddle_model.state_dict()[f'encoder_layers.{i}.attn.q_proj.bias'].numpy()
        k_bias = paddle_model.state_dict()[f'encoder_layers.{i}.attn.k_proj.bias'].numpy()
        v_bias = paddle_model.state_dict()[f'encoder_layers.{i}.attn.v_proj.bias'].numpy()

        # 合并QKV权重 (PyTorch MultiheadAttention需要这种格式)
        # 注意：需要转置，因为PaddlePaddle和PyTorch的线性层权重形状不同
        in_proj_weight = np.concatenate([q_weight.T, k_weight.T, v_weight.T], axis=0)
        in_proj_bias = np.concatenate([q_bias, k_bias, v_bias], axis=0)

        pytorch_state_dict[f'encoder_layers.{i}.attn.in_proj_weight'] = torch.from_numpy(in_proj_weight)
        pytorch_state_dict[f'encoder_layers.{i}.attn.in_proj_bias'] = torch.from_numpy(in_proj_bias)

        # 输出投影层权重也需要转置
        out_proj_weight = paddle_model.state_dict()[f'encoder_layers.{i}.attn.out_proj.weight'].numpy()
        out_proj_bias = paddle_model.state_dict()[f'encoder_layers.{i}.attn.out_proj.bias'].numpy()

        pytorch_state_dict[f'encoder_layers.{i}.attn.out_proj.weight'] = torch.from_numpy(out_proj_weight.T)
        pytorch_state_dict[f'encoder_layers.{i}.attn.out_proj.bias'] = torch.from_numpy(out_proj_bias)

        print(f"✅ 处理注意力层 {i}: QKV权重合并完成")

    # 加载权重到PyTorch模型
    try:
        missing_keys, unexpected_keys = pytorch_model.load_state_dict(pytorch_state_dict, strict=False)

        if missing_keys:
            print(f"⚠️  缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  意外的键: {unexpected_keys}")

        print("✅ 权重加载成功")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return None

    # 保存PyTorch模型 - 使用配置的文件名
    pytorch_model_path = os.path.join(os.path.dirname(paddle_model_path), PYTORCH_MODEL_NAME)
    torch.save({
        'model_state_dict': pytorch_model.state_dict(),
        'config': {
            'image_size': Config.image_size,
            'patch_size': 16,
            'num_classes': Config.num_classes,
            'dim': 384,
            'depth': 6,
            'heads': 8,
            'mlp_ratio': 4,
            'dropout': 0.1
        }
    }, pytorch_model_path)

    print(f"💾 PyTorch模型已保存: {pytorch_model_path}")

    # 验证转换结果
    print("\n" + "=" * 60)
    print("验证转换结果")
    print("=" * 60)

    # 创建相同的测试输入
    np.random.seed(42)
    test_data = np.random.randn(2, 3, Config.image_size, Config.image_size).astype(np.float32)

    # PaddlePaddle推理
    paddle_input = paddle.to_tensor(test_data)
    with paddle.no_grad():
        paddle_output = paddle_model(paddle_input)

    # PyTorch推理
    torch_input = torch.from_numpy(test_data)
    with torch.no_grad():
        torch_output = pytorch_model(torch_input)

    # 比较输出
    paddle_output_np = paddle_output.numpy()
    torch_output_np = torch_output.numpy()

    print(f"Paddle输出形状: {paddle_output_np.shape}")
    print(f"PyTorch输出形状: {torch_output_np.shape}")

    # 计算差异
    diff = np.abs(paddle_output_np - torch_output_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"最大差异: {max_diff:.6f}")
    print(f"平均差异: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("✅ 转换成功！输出基本一致")
    else:
        print("⚠️  输出存在差异，但模型结构已转换完成")

    return pytorch_model


def load_pytorch_model(model_path=None):
    """加载转换后的PyTorch模型"""
    if model_path is None:
        # 使用配置的文件名作为默认路径
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, PYTORCH_MODEL_NAME)

    if not os.path.exists(model_path):
        print(f"❌ 找不到PyTorch模型: {model_path}")
        return None

    # 加载模型配置
    checkpoint = torch.load(model_path)
    config = checkpoint['config']

    # 创建模型
    model = PyTorchViT(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        dim=config['dim'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_ratio=config['mlp_ratio'],
        dropout=config['dropout']
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ 已加载PyTorch模型: {model_path}")
    return model


if __name__ == "__main__":
    # 使用配置的路径直接转换模型
    pytorch_model = convert_paddle_to_pytorch(PADDLE_MODEL_PATH)

    if pytorch_model is not None:
        print("\n🎉 模型转换完成！")
        print(f"转换后的模型已保存为: {PYTORCH_MODEL_NAME}")
        print("您可以使用 load_pytorch_model() 函数加载转换后的模型")

        # 测试加载功能
        loaded_model = load_pytorch_model()
        if loaded_model:
            print("✅ 模型加载测试成功！")