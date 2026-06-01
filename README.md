## DUAD
高迁移图像对抗样本检测算法

[![GitHub](https://img.shields.io/badge/GitHub-yuluoyunyan1949-181717?logo=github)](https://github.com/yuluoyunyan1949)

## 开发环境硬件信息
### GPU
- 型号：NVIDIA GeForce RTX 5060 (Laptop)
- 显存：8151 MiB（约 8 GB）
- CUDA版本：12.8
- 驱动版本：595.97

### CPU
- 型号：Intel Core i9-14900HX
- 物理核心数：16 核（32 线程）
- 虚拟化环境：WSL2

## 项目结构
```markdown
DUAD/
├── checkpoint/                 # 模型权重存放目录
│   ├── vmamba/                 # 无监督训练生成的权重
│   ├── mobileNetV2/            # 高迁移图像对抗样本生成过程中用到的模型定义文件及权重
│   ├── repvgg/                 # 同上
│   ├── resnet20/               # 同上        
│   ├── shufflenetv2/           # 同上
│   └── vgg16_bn/               # 同上
│
├── dataset/
│   └── cifar-10-batches-py/    # cifar10数据集
│
├── result/                     # 实验结果输出目录
│   ├── ablation/
│   ├── compare/
│   └── robustness/
│
├── saved_adv_samples/          # 生成的对抗样本缓存目录
│
├── script/
│   ├── ablation.py             # 消融实验
│   ├── motivation.py           # 方向验证实验
│   ├── robustness.py           # 鲁棒性实验
│   └── run.py                  # 一键脚本
│
├── src/
│   ├── baselines/              # 基线方法实现
│   │   ├── __init__.py
│   │   ├── mse.py              # MSE 重构误差检测
│   │   └── ocsvm.py            # OC-SVM 隐空间异常检测
│   ├── __init__.py
│   ├── attack_generator.py     # 高迁移图像对抗样本生成器
│   ├── data_loader.py          # CIFAR-10加载与预处理器
│   ├── detector.py             # 图像对抗样本检测器
│   ├── encoder.py              # DualDomainAE 双域自编码器模型定义
│   ├── gui.py                  # 用户操作页面（Gradio实现）
│   └── trainer.py              # 无监督训练器 & 检测阈值初始化
│
├── VMamba                      # 克隆的VMamba仓库（有改动）
├── README.md
├── requirements.txt
└── simhei.ttf                  # 绘图时使用的中文字体文件
```

## 使用方式
参考 DUAD/script/ 目录下各文件的内容
