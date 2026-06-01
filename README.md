## DUAD
高迁移图像对抗样本检测算法

[![GitHub](https://img.shields.io/badge/GitHub-yuluoyunyan1949-181717?logo=github)](https://github.com/yuluoyunyan1949)

## 开发环境
### GPU
- 型号：NVIDIA GeForce RTX 5060 (Laptop)
- 显存：8151 MiB（约 8 GB）
- CUDA版本：12.8.61
- 驱动版本：595.97

### CPU
- 型号：Intel Core i9-14900HX
- 物理核心数：16 核（32 线程）
- 虚拟化环境：WSL2

### 其他
- python 3.10.20
- g++ 13.3.0


## 项目结构
```markdown
DUAD/
├── checkpoint/                     # 模型权重存放目录
│   ├── vmamba/                     # 无监督训练生成的权重
│   │    ├── best_model.pth         # 第二阶段最佳训练权重（最终使用这个）
│   │    └── spatial_only_best.pth  # 第一阶段最佳训练权重
│   │
│   ├── mobileNetV2/                # 生成高迁移图像对抗集用到的模型（下面4个也是）
│   │    ├── mfiae_model.pt         # 模型权重文件
│   │    └── model.py               # 模型定义文件
│   ├── repvgg/
│   │    ├── mfiae_model.pt
│   │    └── model.py
│   ├── resnet20/
│   │    ├── mfiae_model.pt
│   │    └── model.py
│   ├── shufflenetv2/
│   │    ├── mfiae_model.pt
│   │    └── model.py
│   └── vgg16_bn/
│        ├── mfiae_model.pt
│        └── model.py
│
├── dataset/
│   └── cifar-10-batches-py/        # cifar10数据集
│
├── result/                         # 实验结果输出目录
│   ├── ablation/
│   ├── compare/
│   └── robustness/
│
├── saved_adv_samples/              # 生成的对抗样本缓存目录
│
├── script/
│   ├── ablation.py                 # 消融实验
│   ├── motivation.py               # 方向验证实验
│   ├── robustness.py               # 鲁棒性实验
│   └── run.py                      # 一键脚本
│
├── src/
│   ├── baselines/                  # 基线方法实现
│   │   ├── __init__.py
│   │   ├── mse.py                  # MSE 重构误差检测
│   │   └── ocsvm.py                # OC-SVM 隐空间异常检测
│   ├── __init__.py
│   ├── attack_generator.py         # 高迁移图像对抗样本生成器
│   ├── data_loader.py              # CIFAR-10加载与预处理器
│   ├── detector.py                 # 图像对抗样本检测器
│   ├── encoder.py                  # DualDomainAE 双域自编码器模型定义
│   ├── gui.py                      # 用户操作页面（Gradio实现）
│   └── trainer.py                  # 无监督训练器 & 检测阈值初始化
│
├── VMamba                          # 克隆的VMamba仓库（有改动）
├── README.md
├── requirements.txt
└── simhei.ttf                      # 绘图时使用的中文字体文件
```

## Start
### 一、安装
#### 1.克隆仓库：
```bash
git clone https://github.com/yuluoyunyan1949/DUAD.git
cd DUAD
```
#### 2.创建虚拟环境
```bash
conda create -n duad python=3.10
conda activate duad
```


#### 3.安装依赖
```bash
pip install -r requirements.txt
```

#### 4.编译 CUDA 扩展
```bash
// 若 pip install 失败，请手动进入 VMamba/kernels/selective_scan 执行
pip install -e .
```

### 二、补全项目结构
- 根据README.md的项目结构部分手动补全部分目录（如saved_adv_samples），因为Github不支持上传空目录。
- 检查cifar10数据集是否存在。若无，可在 https://www.cs.toronto.edu/~kriz/cifar.html 下载并解压在项目结构约定的位置
- 检查模型定义文件及权重是否存在。若无，可在 https://github.com/chenyaofo/pytorch-cifar-models 下载，并将定义文件统一更名为“model.py”，权重文件统一更名为“mfiae_model.pt”保存在各自的目录中

### 三、训练vmamba权重
```bash
# 假设目前终端目前位于DUAD根目录，执行聚合脚本run即可开始训练
python script/run.py
```

### 四、使用
- script/目录下各脚本都可以直接执行
- 若权重已训练完成，再次执行run脚本可打开图形操作界面
