## DUAD
高迁移图像对抗样本检测算法

## 项目结构
DUAD/
├── checkpoint/                 # 模型权重存放目录
│   ├── mobileNetV2/            # 源模型权重 
│   ├── vmamba/                 # 双域自编码器模型权重
│   ├── repvgg/                 # 源模型权重
│   ├── resnet20/               # 源模型权重          
│   ├── shufflenetv2/           # 目标模型权重
│   └── vgg16_bn/               # 目标模型权重
│
├── dataset/                    # 实验结果输出
│   └── cifar-10-batches-py/    # cifar10数据集
│
├── result/                     # 实验结果输出目录
│   ├── ablation/               # 消融实验
│   ├── compare/                # 结构改进前后对比实验
│   └── robustness/             # 鲁棒性实验
│
├── saved_adv_samples/          # 生成的对抗样本缓存
│
├── script/                     # 实验结果输出
│   ├── ablation.py             # 消融实验
│   ├── motivation.py           # 方向验证实验
│   ├── robustness.py           # 鲁棒性实验
│   └── run.py                  # 一键脚本
│
├── src/                        # 核心源码目录
│   ├── baselines/              # 基线方法实现
│   │   ├── __init__.py
│   │   ├── mse.py              # MSE 重构误差检测
│   │   └── ocsvm.py            # OC-SVM 隐空间异常检测
│   ├── __init__.py
│   ├── attack_generator.py     # 对抗样本生成工具
│   ├── data_loader.py          # CIFAR-10 数据加载与预处理
│   ├── detector.py             # 空域+频域双域检测逻辑实现
│   ├── encoder.py              # DualDomainAE 双域自编码器模型定义
│   ├── gui.py                  # 用户操作页面
│   └── trainer.py              # 模型训练与阈值初始化
│
├── VMamba                      # 克隆的VMamba仓库（改动版）
├── README.md
├── requirements.txt            # 依赖管理
└── simhei.ttf                  # 绘图时使用的中文字体文件


**如果是在不支持等宽字体的地方（比如某些聊天软件）**，树形图可能会乱掉。那时可以改用**无序列表嵌套**，兼容性最强。例如：

```markdown
- DUAD/
  - checkpoint/
    - mobileNetV2/
    - vmamba/
    - repvgg/
    - resnet20/
    - shufflenetv2/
    - vgg16_bn/
  - dataset/
    - cifar-10-batches-py/
  - result/
    - ablation/
    - compare/
    - robustness/
  - saved_adv_samples/
  - script/
    - ablation.py
    - motivation.py
    - robustness.py
    - run.py
  - src/
    - baselines/
      - __init__.py
      - mse.py
      - ocsvm.py
    - __init__.py
    - attack_generator.py
    - data_loader.py
    - detector.py
    - encoder.py
    - gui.py
    - trainer.py
  - VMamba
  - README.md
  - requirements.txt
  - simhei.ttf

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
