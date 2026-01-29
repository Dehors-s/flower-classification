# Flower Classification Project

## 项目简介
本项目是一个基于深度学习的花卉图像分类系统。项目使用了 Vision Transformer (ViT) 等先进模型进行训练和预测。代码结构清晰，包含了模型定义、训练脚本、评估工具以及相关的数据处理工具。

主要目录结构如下：
- `models/`: 包含 ViT 模型定义及 PaddlePaddle 到 PyTorch 的转换脚本。
- `training_scripts/`: 包含不同版本的训练脚本 (train_v1.py - train_v5.py)。
- `evaluation/`: 包含模型评估、检测精度计算及预测脚本。
- `utils/`: 包含图像增强、数据集建立等实用工具。
- `results/`: 存放训练结果和历史记录。

## 环境安装

1. 克隆本项目到本地。
2. 确保已安装 Python 环境 (建议 Python 3.8+)。
3. 安装项目依赖库：

```bash
pip install -r requirements.txt
```

**依赖库说明：**
- 核心深度学习框架：`paddlepaddle`, `torch`
- 图像处理与数据分析：`numpy`, `pandas`, `opencv-python`, `Pillow`, `matplotlib`
- 其他工具：`tqdm`, `rasterio`

## 训练步骤

本项目提供了多个版本的训练脚本，位于 `training_scripts/` 目录下。

1. **准备数据**
   确保数据集已按照要求准备好（参考 `utils/数据集建立.py`）。

2. **开始训练**
   选择一个训练脚本开始训练，例如使用 `train_v1.py`：

   ```bash
   python training_scripts/train_v1.py
   ```

   或者使用最新的训练脚本：

   ```bash
   python training_scripts/train_v5.py
   ```

3. **模型评估**
   训练完成后，可以使用 `evaluation/` 目录下的脚本评估模型性能：

   ```bash
   python evaluation/检测精度.py
   ```

4. **查看结果**
   训练过程中的日志和结果文件将保存在 `results/` 目录或脚本指定的输出目录中。
